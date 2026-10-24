use crate::Tile;

pub const WIDE: Tile = Tile { bm: 256, bn: 128, wn: 2, stages: 2 };
pub const SQUARE: Tile = Tile { bm: 128, bn: 128, wn: 4, stages: 2 };
pub const THIN: Tile = Tile { bm: 128, bn: 16, wn: 1, stages: 2 };
pub const SMALL: Tile = Tile { bm: 64, bn: 64, wn: 2, stages: 2 };
