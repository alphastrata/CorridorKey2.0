fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let paths: Vec<std::path::PathBuf> = std::env::args_os().skip(1).map(std::path::PathBuf::from).map(|p| std::path::absolute(&p).unwrap_or(p)).collect();
    ck_gui::run(&paths)?;
    Ok(())
}
