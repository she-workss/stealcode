//! Micro-benchmark: где сидит время? Запуск: cargo run --release -p
//! voice --example bench -- <gguf>
use std::time::Instant;

use voice::nemotron::{Nemotron, weights::Lin};

fn main() {
    let path = std::env::args().nth(1).expect("usage: bench <gguf>");
    let mut m = Nemotron::load(std::path::Path::new(&path)).unwrap();
    let mut scratch = Vec::new();
    let mut dequant = std::time::Duration::ZERO;
    let mut dq_elems: usize = 0;
    for b in &mut m.encoder.blocks {
        for lin in [
            &mut b.ff1_lin1,
            &mut b.ff1_lin2,
            &mut b.attn_q,
            &mut b.attn_k,
            &mut b.attn_v,
            &mut b.attn_pos,
            &mut b.attn_out,
            &mut b.pw1,
            &mut b.pw2,
            &mut b.ff2_lin1,
            &mut b.ff2_lin2,
        ] {
            let Some(q) = &lin.q else { continue };
            let n = q.rows() * q.row_len();
            let s = Instant::now();
            q.to_f32(&mut scratch);
            dequant += s.elapsed();
            dq_elems += n;
        }
    }
    eprintln!("dequant всех Lin энкодера: {dequant:?}, {dq_elems} элементов");

    fn best_time(mut f: impl FnMut() -> usize) -> f64 {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let s = Instant::now();
            f();
            best = best.min(s.elapsed().as_secs_f64());
        }
        best
    }

    // Полный encode на синтетическом меле 81 фрейм (6.49 с).
    let n_mels = m.cfg.preprocessor.n_mels;
    let mel = vec![0.01f32; 81 * n_mels];
    let mut enc = Vec::new();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let s = Instant::now();
        let t_enc = m.encoder.encode(&mel, 81, Some(101), &mut enc).unwrap();
        best = best.min(s.elapsed().as_secs_f64());
        eprintln!("t_enc={t_enc} enc_len={}", enc.len());
    }
    eprintln!("encode 81 фреймов (мин из 5): {best:.3}s");

    // Сам decoder.decode().
    let mut best_d = f64::MAX;
    for _ in 0..3 {
        let s = Instant::now();
        let tokens = m.decoder.decode(&enc, 11).unwrap();
        best_d = best_d.min(s.elapsed().as_secs_f64());
        std::hint::black_box(&tokens);
    }
    eprintln!("decode 11 фреймов (мин из 3): {best_d:.4}s");

    // Реальный размер audio.wav: 650 mel-фреймов (81 кадр энкодера).
    let mel2 = vec![0.01f32; 650 * n_mels];
    let mut enc2 = Vec::new();
    let t2 = m.encoder.encode(&mel2, 650, Some(101), &mut enc2).unwrap();
    eprintln!(
        "encode 650 mel (81 кадр): {:.3}s (t_enc={t2})",
        best_time(|| {
            m.encoder.encode(&mel2, 650, Some(101), &mut enc2).unwrap()
        })
    );
    eprintln!("decode 81 кадр: {:.4}s", {
        let mut b = f64::MAX;
        for _ in 0..3 {
            let s = Instant::now();
            let tok = m.decoder.decode(&enc2, t2).unwrap();
            std::hint::black_box(&tok);
            b = b.min(s.elapsed().as_secs_f64());
        }
        b
    });

    // Декодирование: время предсказателя (LSTM) против joint.
    let mut dec_t = std::time::Duration::ZERO;
    let mut joint_t = std::time::Duration::ZERO;
    let mut lstm_steps = 0usize;
    let mut joint_calls = 0usize;
    let hidden = m.decoder.predictor.hidden;
    let n_layers = m.decoder.predictor.layers.len();
    let mut h = vec![vec![0.0f32; hidden]; n_layers];
    let mut c = vec![vec![0.0f32; hidden]; n_layers];
    let mut nh = vec![vec![0.0f32; hidden]; n_layers];
    let mut nc = vec![vec![0.0f32; hidden]; n_layers];
    let mut embed_x = vec![0.0f32; hidden];
    let mut pred_proj = vec![0.0f32; m.decoder.joint.joint_h];
    let dec_idx = n_layers - 1;
    for k in 0..200 {
        let s = Instant::now();
        m.decoder.predictor.step(
            (k % 300) as i32,
            &h,
            &c,
            &mut nh,
            &mut nc,
            &mut embed_x,
        );
        dec_t += s.elapsed();
        lstm_steps += 1;
        let s = Instant::now();
        m.decoder.joint.pred.matvec(&nh[dec_idx], &mut pred_proj);
        joint_t += s.elapsed();
        joint_calls += 1;
        std::hint::black_box(&pred_proj);
    }
    eprintln!(
        "LSTM step x{lstm_steps}: {dec_t:?} ({}мс/шаг)",
        dec_t.as_millis() as f64 / lstm_steps as f64
    );
    eprintln!(
        "joint.pred matvec x{joint_calls}: {joint_t:?} ({}мс/вызов)",
        joint_t.as_millis() as f64 / joint_calls as f64
    );

    // f32 matvec против q8 matvec на одинаковых данных.
    let lin = &m.encoder.blocks[0].ff1_lin1;
    let x = vec![0.5f32; lin.inp];
    let mut y = vec![0.0f32; lin.out];
    let mut t = Instant::now();
    for _ in 0..3 {
        lin.matvec(&x, &mut y);
    }
    eprintln!("q8 matvec ff1_lin1: {:?}", t.elapsed() / 3);
    if let Some(q) = &lin.q {
        let mut f = Vec::new();
        q.to_f32(&mut f);
        let flin = Lin {
            q: None,
            f: Some(f),
            bias: None,
            out: lin.out,
            inp: lin.inp,
        };
        t = Instant::now();
        for _ in 0..3 {
            flin.matvec(&x, &mut y);
        }
        eprintln!("f32 matvec ff1_lin1: {:?}", t.elapsed() / 3);
    }

    // Пропускная способность forward_t (sgemm) от размера батча t.
    let mut scratch = Vec::new();
    for t_n in [11usize, 83, 375, 1024] {
        let xt = vec![0.1f32; lin.inp * t_n];
        let mut y2 = Vec::new();
        let s = Instant::now();
        for _ in 0..3 {
            lin.forward_t(&mut scratch, &xt, t_n, &mut y2);
        }
        let per = s.elapsed() / 3;
        let flop = 2.0 * lin.out as f64 * lin.inp as f64 * t_n as f64;
        eprintln!(
            "ff1 forward_t t={t_n}: {per:?} ({:.0} GFLOP/s)",
            flop / per.as_secs_f64() / 1e9
        );
    }
}
