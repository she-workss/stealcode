use clap::Parser;

#[derive(Parser, Clone, Debug)]
pub(crate) struct CliOptions {
    /// Agent to use.
    #[arg(long)]
    pub agent: Option<String>,

    /// Continue the last session.
    #[arg(short = 'c', long = "continue")]
    pub r#continue: bool,

    /// Additional domains to allow for CORS.
    #[arg(long)]
    pub cors: Vec<String>,

    /// Fork the session when continuing.
    #[arg(long)]
    pub fork: bool,

    /// Hostname to listen on.
    #[arg(long, default_value = "127.0.0.1")]
    pub hostname: String,

    /// Enable mDNS service discovery (defaults hostname to 0.0.0.0).
    #[arg(long, default_value_t = false)]
    pub mdns: bool,

    /// Custom domain name for mDNS service.
    #[arg(long, default_value = "stealcode.local")]
    pub mdns_domain: String,

    /// Model to use in the format of provider/model.
    #[arg(short, long)]
    pub model: Option<String>,

    /// Port to listen on.
    #[arg(long, default_value_t = 0)]
    pub port: u16,

    /// Print logs to stderr.
    #[arg(long)]
    pub print_logs: bool,

    /// Prompt to use.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Run without external plugins.
    #[arg(long)]
    pub pure: bool,

    /// Session id to continue.
    #[arg(short, long)]
    pub session: Option<String>,
}
