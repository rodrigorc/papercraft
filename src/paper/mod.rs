mod model;
mod craft;

pub use model::*;
pub use craft::*;

use crate::util_3d::*;
use serde::{Serialize, Deserialize, ser::{SerializeStruct, SerializeSeq}};

mod ser {
    use super::*;
    pub mod vector2 {
        use super::*;
        pub fn serialize<S>(data: &Vector2, serializer: S) -> Result<S::Ok, S::Error>
            where S: serde::Serializer
        {
            let mut seq = serializer.serialize_seq(Some(3))?;
            seq.serialize_element(&data.x)?;
            seq.serialize_element(&data.y)?;
            seq.end()
        }
        pub fn deserialize<'de, D>(deserializer: D) -> Result<Vector2, D::Error>
            where D: serde::Deserializer<'de>
        {
            let data = <[f32; 2]>::deserialize(deserializer)?;
            Ok(Vector2::from(data))
        }
    }
    pub mod vector3 {
        use super::*;
        pub fn serialize<S>(data: &Vector3, serializer: S) -> Result<S::Ok, S::Error>
            where S: serde::Serializer
        {
            let mut seq = serializer.serialize_seq(Some(3))?;
            seq.serialize_element(&data.x)?;
            seq.serialize_element(&data.y)?;
            seq.serialize_element(&data.z)?;
            seq.end()
        }
        pub fn deserialize<'de, D>(deserializer: D) -> Result<Vector3, D::Error>
            where D: serde::Deserializer<'de>
        {
            let data = <[f32; 3]>::deserialize(deserializer)?;
            Ok(Vector3::from(data))
        }
    }
    // Beware! This serializes pnly the values, not the keys.
    pub mod slot_map {
        use super::*;
        pub fn serialize<K, V, S>(data: &slotmap::SlotMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
            where S: serde::Serializer,
                  K: slotmap::Key,
                  V: Serialize,
        {
            let mut seq = serializer.serialize_seq(Some(data.len()))?;
            for (_, d) in data {
                seq.serialize_element(d)?;
            }
            seq.end()
        }
        pub fn deserialize<'de, D, K, V>(deserializer: D) -> Result<slotmap::SlotMap<K, V>, D::Error>
            where D: serde::Deserializer<'de>,
                  K: slotmap::Key,
                  V: Deserialize<'de>,
        {
            let data = <Vec<V>>::deserialize(deserializer)?;
            let mut map = slotmap::SlotMap::with_key();
            for d in data {
                map.insert(d);
            }
            Ok(map)
        }
    }
    pub mod matrix3 {
        use cgmath::Rad;

        use super::*;
        pub fn serialize<S>(data: &Matrix3, serializer: S) -> Result<S::Ok, S::Error>
            where S: serde::Serializer
        {
            let mut map = serializer.serialize_struct("Island", 3)?;
            let r = data[0][1].atan2(data[0][0]);
            map.serialize_field("x", &data[2][0])?;
            map.serialize_field("y", &data[2][1])?;
            map.serialize_field("r", &r)?;
            map.end()
        }
        pub fn deserialize<'de, D>(deserializer: D) -> Result<Matrix3, D::Error>
            where D: serde::Deserializer<'de>
        {
            #[derive(Deserialize)]
            struct Def { x: f32, y: f32, r: f32 }
            let d = Def::deserialize(deserializer)?;
            let r = Matrix3::from(Matrix2::from_angle(Rad(d.r)));
            let t = Matrix3::from_translation(Vector2::new(d.x, d.y));
            Ok(t * r)
        }
    }

}
