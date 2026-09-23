mod craft;
mod model;

pub use craft::*;
pub use model::*;

use crate::util_3d::*;
use serde::{
    Deserialize, Serialize,
    ser::{SerializeSeq, SerializeStruct},
};
mod ser {
    use super::*;
    pub mod vector2 {
        use super::*;
        pub fn serialize<S>(data: &Vector2, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let mut seq = serializer.serialize_seq(Some(3))?;
            seq.serialize_element(&data.x)?;
            seq.serialize_element(&data.y)?;
            seq.end()
        }
        pub fn deserialize<'de, D>(deserializer: D) -> Result<Vector2, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let data = <[f32; 2]>::deserialize(deserializer)?;
            Ok(Vector2::from(data))
        }
    }
    pub mod vector3 {
        use super::*;
        pub fn serialize<S>(data: &Vector3, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let mut seq = serializer.serialize_seq(Some(3))?;
            seq.serialize_element(&data.x)?;
            seq.serialize_element(&data.y)?;
            seq.serialize_element(&data.z)?;
            seq.end()
        }
        pub fn deserialize<'de, D>(deserializer: D) -> Result<Vector3, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let data = <[f32; 3]>::deserialize(deserializer)?;
            Ok(Vector3::from(data))
        }
    }
    // Beware! This serializes only the values, not the keys.
    pub mod slot_map {
        pub trait SlotMapKeyOrder {
            type OrderKey: Ord;
            fn slot_map_key_order(&self) -> Self::OrderKey;
            fn slot_map_set_key_index(&mut self, index: usize);
        }

        use super::*;
        pub fn serialize<K, V, S>(
            data: &slotmap::SlotMap<K, V>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
            K: slotmap::Key,
            V: Serialize + SlotMapKeyOrder,
        {
            let mut seq = serializer.serialize_seq(Some(data.len()))?;
            let mut values = data.values().collect::<Vec<&V>>();
            values.sort_by_key(|v| v.slot_map_key_order());
            for d in values {
                seq.serialize_element(d)?;
            }
            seq.end()
        }
        pub fn deserialize<'de, D, K, V>(
            deserializer: D,
        ) -> Result<slotmap::SlotMap<K, V>, D::Error>
        where
            D: serde::Deserializer<'de>,
            K: slotmap::Key,
            V: Deserialize<'de> + SlotMapKeyOrder,
        {
            let data = <Vec<V>>::deserialize(deserializer)?;
            let mut map = slotmap::SlotMap::with_key();
            for (i, mut d) in data.into_iter().enumerate() {
                d.slot_map_set_key_index(i);
                map.insert(d);
            }
            Ok(map)
        }
    }
}
