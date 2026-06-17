use clap::Subcommand;
use clap_complete::Shell;

#[derive(Subcommand)]
pub(crate) enum CliCommands {
    /// Start ACP (Agent Client Protocol) server.
    Acp,

    /// Attach to a running stealcode server.
    Attach {
        /// Url of the running server.
        url: String,
    },

    /// Manage AI providers and credentials.
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },

    /// Generate shell completion script.
    Completion {
        /// Shell to generate completions for.
        shell: Shell,
    },

    /// Debugging and troubleshooting tools.
    Debug,

    /// Start the native desktop GUI.
    #[cfg(feature = "desktop")]
    Desktop,

    /// Database tools.
    Db,

    /// Export session data as JSON.
    Export {
        /// Session id to export.
        session_id: Option<String>,
    },

    /// Manage GitHub agent.
    Github,

    /// Import session data from JSON file or URL.
    Import {
        /// File path or URL to import from.
        file: String,
    },

    /// Manage MCP (Model Context Protocol) servers.
    Mcp,

    /// List all available models.
    Models {
        /// Provider to list models for.
        provider: Option<String>,
    },

    /// Install plugin and update config.
    #[command(alias = "plug")]
    Plugin {
        /// Plugin module to install.
        module: String,
    },

    /// Fetch and checkout a GitHub PR branch, then run stealcode.
    Pr {
        /// PR number.
        number: u64,
    },

    /// Manage AI providers and credentials.
    Providers,

    /// Run stealcode with a message.
    Run {
        /// Message to send.
        message: Vec<String>,
    },

    /// Manage sessions.
    Session,

    /// Starts a headless stealcode server.
    #[cfg(feature = "server")]
    Serve,

    /// Show token usage and cost statistics.
    Stats,

    /// Uninstall stealcode and remove all related files.
    Uninstall,

    /// Upgrade stealcode to the latest or a specific version.
    Upgrade {
        /// Target version to upgrade to.
        target: Option<String>,
    },

    /// Start stealcode server and open web interface.
    #[cfg(feature = "web")]
    Web,
}

#[derive(Subcommand)]
pub(crate) enum AuthCommands {
    /// List providers and credentials.
    #[command(alias = "ls")]
    List,

    /// Log in to a provider.
    Login {
        /// URL of the provider to log in to.
        url: Option<String>,
    },

    /// Log out from a configured provider.
    Logout,
}
