package base

import (
   "fmt"
)

// Feature types are very thin (no memory footprint) wrappers over actual data with some useful methods.
// Instead of defining the features in the metadata for a tuple/collection,
// we will put a very thin wrapper around every piece of data.

type Feature interface {
   Value() interface{}
   IsNumeric() bool
}

type NumericFeature interface {
   Feature
   NumericValue() float64
}

// Nil

type NilFeature byte

func Nil() NilFeature {
   return NilFeature(0);
}

func (this NilFeature) Value() interface{} {
   return nil;
}

func (this NilFeature) IsNumeric() bool {
   return false;
}

// Integer

type IntFeature int

func Int(val int) IntFeature {
   return IntFeature(val);
}

func (this IntFeature) Value() interface{} {
   return int(this);
}

func (this IntFeature) NumericValue() float64 {
   return float64(this);
}

func (this IntFeature) IntValue() int {
   return int(this);
}

func (this IntFeature) IsNumeric() bool {
   return true;
}

// Float

type FloatFeature float64

func Float(val float64) FloatFeature {
   return FloatFeature(val);
}

func (this FloatFeature) Value() interface{} {
   return float64(this);
}

func (this FloatFeature) NumericValue() float64 {
   return float64(this);
}

func (this FloatFeature) FloatValue() float64 {
   return float64(this);
}

func (this FloatFeature) IsNumeric() bool {
   return true;
}

// Bool

type BoolFeature bool

func Bool(val bool) BoolFeature {
   return BoolFeature(val);
}

func (this BoolFeature) Value() interface{} {
   return bool(this);
}

func (this BoolFeature) NumericValue() float64 {
   if (bool(this)) {
      return 1;
   }
   return 0;
}

func (this BoolFeature) BoolValue() bool {
   return bool(this);
}

func (this BoolFeature) IsNumeric() bool {
   return true;
}

// String

type StringFeature string

func String(val string) StringFeature {
   return StringFeature(val);
}

func (this StringFeature) Value() interface{} {
   return string(this);
}

func (this StringFeature) StringValue() string {
   return string(this);
}

func (this StringFeature) IsNumeric() bool {
   return false;
}

// Infer feature types from the data.
// If the data is already a feature type, then that will be returned.
// If the data type does not match the allowed feature types
// (eg a string is passed to InferNumericFeature(), then we will panic.

func InferFeatures(data []interface{}) []Feature {
   var rtn []Feature = make([]Feature, len(data));
   for i, val := range(data) {
      rtn[i] = InferFeature(val);
   }
   return rtn;
}

func InferFeature(data interface{}) Feature {
   if (data == nil) {
      return Nil();
   }

   // Note that we can't group up the cases (fallthrough semantics)
   // since then data would be an interface{} instead of a hard type.
   // Which would then forace a type assertion before the cast.
   switch data := data.(type) {
   // Check for Feature types first.
   case NilFeature:
      return data;
   case IntFeature:
      return data;
   case FloatFeature:
      return data;
   case StringFeature:
      return data;
   case BoolFeature:
      return data;
   // Check for builtin types.
   case int:
      return Int(data)
   case int32:
      return Int(int(data))
   case int64:
      return Int(int(data))
   case uint:
      return Int(int(data))
   case uint32:
      return Int(int(data))
   case uint64:
      return Int(int(data))
   case bool:
      return Bool(data);
   case float32:
      return Float(float64(data))
   case float64:
      return Float(data)
   case string:
      return String(data);
   default:
      panic(fmt.Sprintf("Unknown type for feature conversion: %T", data))
   }
}

func InferNumericFeatures(data []interface{}) []NumericFeature {
   var rtn []NumericFeature = make([]NumericFeature, len(data));
   for i, val := range(data) {
      rtn[i] = InferNumericFeature(val);
   }
   return rtn;
}

func InferNumericFeature(data interface{}) NumericFeature {
   var feature Feature = InferFeature(data);
   numericFeature, ok := feature.(NumericFeature);
   if (!ok) {
      panic(fmt.Sprintf("Tried to infer numeric feature from non-numeric value: %v", data));
   }

   return numericFeature;
}

func InferIntFeatures(data []interface{}) []IntFeature {
   var rtn []IntFeature = make([]IntFeature, len(data));
   for i, val := range(data) {
      rtn[i] = InferIntFeature(val);
   }
   return rtn;
}

func InferIntFeature(data interface{}) IntFeature {
   var feature Feature = InferFeature(data);
   intFeature, ok := feature.(IntFeature);
   if (!ok) {
      panic(fmt.Sprintf("Tried to infer int feature from non-int value: %v", data));
   }

   return intFeature;
}
