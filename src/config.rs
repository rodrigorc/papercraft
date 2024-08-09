use anyhow::Result;
use serde::*;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub locale: String,
    pub light_mode: bool,
}

impl Config {
    fn file_name() -> Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("com", "rodrigorc", "papercraft")
            .ok_or(anyhow::anyhow!("Unknown configuration directory"))?;
        let dir = dirs.preference_dir();
        Ok(PathBuf::from(dir).join("papercraft.json"))
    }
    fn load() -> Result<Config> {
        let file_name = Self::file_name()?;
        let f = std::fs::File::open(file_name)?;
        let f = std::io::BufReader::new(f);
        let cfg = serde_json::from_reader(f)?;
        Ok(cfg)
    }
    pub fn save(&self) -> Result<()> {
        let file_name = Self::file_name()?;
        if let Some(d) = file_name.parent() {
            std::fs::create_dir_all(d)?
        }
        let f = std::fs::File::create(file_name)?;
        let f = std::io::BufWriter::new(f);
        serde_json::to_writer(f, self)?;
        Ok(())
    }

    pub fn load_or_default() -> Config {
        if let Ok(c) = Self::load() {
            return c;
        }
        let locale = sys_locale::get_locale().unwrap_or(String::from("en"));
        let locale = locale.split('-').next().unwrap().to_owned();
        Config {
            locale,
            light_mode: false,
        }
    }
}
