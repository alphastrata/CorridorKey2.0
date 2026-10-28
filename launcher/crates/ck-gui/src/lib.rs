slint::include_modules!();

mod app;
mod clips;
mod command;
pub mod convert;
mod frames;
mod log;
mod plots;
mod runner;
mod settings;
mod telemetry;

pub use app::run;
