use std::env;
use std::process::{Command, ExitStatus};

fn main() {
    // Collect the arguments passed to xtask (e.g., "ci" or "check-serde")
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("ci");

    match command {
        "ci" => run_ci_pipeline(),
        "check-msrv" => run_msrv_check(),
        _ => {
            eprintln!("❌ Unknown xtask command: '{}'", command);
            eprintln!("Available commands: ci, check-serde");
            std::process::exit(1);
        }
    }
}

fn run_ci_pipeline() {
    let tasks: &[(&str, &[&str])] = &[
        ("fmt", &["--check"]),
        ("clippy", &["--all-targets", "--", "-D", "warnings"]),
        ("test", &[]),
        ("doc", &["--no-deps"]),
        ("publish", &["--dry-run"]),
    ];

    for (subcommand, args) in tasks {
        println!("🚀 Running: cargo {} {}", subcommand, args.join(" "));
        let status = Command::new("cargo").arg(subcommand).args(*args).status();
        check_status(status, subcommand);
    }
    println!("✅ Local CI pipeline completed successfully!");
}

fn run_msrv_check() {
    println!("🚀 Running: cargo +1.89.0 check --lib");

    // We invoke cargo, passing the toolchain string as the very first argument
    let status = Command::new("cargo").arg("+1.89.0").arg("check").arg("--lib").status();

    check_status(status, "check-msrv");
    println!("✅ MSRV 1.89 check passed!");
}

fn check_status(status: Result<ExitStatus, std::io::Error>, name: &str) {
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("❌ Error: '{}' failed.", name);
            std::process::exit(s.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("Fatal error: Could not invoke command for '{}': {}", name, e);
            std::process::exit(1);
        }
    }
}
