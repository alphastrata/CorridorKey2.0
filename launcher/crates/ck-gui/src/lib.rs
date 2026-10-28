slint::include_modules!();

mod app;
mod clips;
mod command;
pub mod convert;
#[cfg(unix)]
mod daemon;
mod frames;
mod log;
mod plots;
mod runner;
mod settings;
mod telemetry;

pub use app::run;
