use clap::Parser;
use fs::{daemon, update, Commands, Opt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::parse();

    match opt.commands {
        Commands::Run(args) => daemon::run(args)?,
        #[cfg(target_family = "unix")]
        Commands::Start(args) => daemon::start(args)?,
        #[cfg(target_family = "unix")]
        Commands::Restart(args) => daemon::restart(args)?,
        #[cfg(target_family = "unix")]
        Commands::Stop => daemon::stop()?,
        #[cfg(target_family = "unix")]
        Commands::Log => daemon::log()?,
        #[cfg(target_family = "unix")]
        Commands::PS => daemon::status()?,
        Commands::Update => update::update()?,
    };

    Ok(())
}
