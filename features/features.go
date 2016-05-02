package features

type FeatureType int

const (
   Numeric FeatureType = iota
   Boolean
   String
)

type Features []FeatureType
