mod data;
mod exporter;
mod importer;

// glTF seems to store positions in meters, but we use millimeters.
const GLTF_SCALE: f32 = 1000.0;

pub use exporter::{GltfFormat, export};
pub use importer::GltfImporter;
