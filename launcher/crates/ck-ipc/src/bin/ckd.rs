/// Per-user daemon that runs `ck` jobs for the GUI and other clients.
#[derive(argh::FromArgs)]
struct Args {}

fn main() -> Result<(), ck_ipc::Error> {
    let _: Args = argh::from_env();
    ck_ipc::server::serve()
}
