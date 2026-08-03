fn main() {
    for path in std::env::args().skip(1) {
        match dominion_ai::example::read_shard(&path) {
            Ok(v) => {
                let td = v
                    .iter()
                    .filter(|e| (e.td_target - e.outcome).abs() > 1e-3)
                    .count();
                let name = path.rsplit('/').next().unwrap_or(&path).to_string();
                println!(
                    "{}: {} examples, {} with distinct TD targets",
                    name,
                    v.len(),
                    td
                );
            }
            Err(e) => println!("{}: FAILED - {}", path, e),
        }
    }
}
