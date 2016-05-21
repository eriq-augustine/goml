package base

import (
   "reflect"
)

type Tuple interface {
   GetData(index int) Feature
   SetData(index int, newValue interface{})
   GetClass() Feature
   DataSize() int
}

type NumericTuple interface {
   Tuple
   GetNumericData(index int) float64
}

type IntTuple interface {
   NumericTuple
   GetIntData(index int) int
}

// General

type GeneralTuple struct {
   Data []Feature
   Class Feature
}

// Feature types will be inferred.
func NewTuple(data []interface{}, class interface{}) Tuple {
   return GeneralTuple{InferFeatures(data), InferFeature(class)};
}

func (this GeneralTuple) GetData(index int) Feature {
   return this.Data[index];
}

func (this GeneralTuple) SetData(index int, newValue interface{}) {
   this.Data[index] = InferFeature(newValue);
}

func (this GeneralTuple) GetClass() Feature {
   return this.Class;
}

func (this GeneralTuple) DataSize() int {
   return len(this.Data);
}

// Numeric

type FloatTuple struct {
   Data []NumericFeature
   Class Feature
}

// Feature types will be inferred.
func NewNumericTuple(data []interface{}, class interface{}) NumericTuple {
   return FloatTuple{InferNumericFeatures(data), InferFeature(class)};
}

func (this FloatTuple) GetData(index int) Feature {
   return this.Data[index];
}

func (this FloatTuple) SetData(index int, newValue interface{}) {
   this.Data[index] = InferNumericFeature(newValue);
}

func (this FloatTuple) GetNumericData(index int) float64 {
   return this.Data[index].NumericValue();
}

func (this FloatTuple) GetClass() Feature {
   return this.Class;
}

func (this FloatTuple) DataSize() int {
   return len(this.Data);
}

// Integer
// Note: the name is pretty bad, but we want people passing the interface not struct.

type IntegerTuple struct {
   Data []IntFeature
   Class Feature
}

// Feature types will be inferred.
func NewIntTuple(data []interface{}, class interface{}) IntTuple {
   return IntegerTuple{InferIntFeatures(data), InferFeature(class)};
}

func (this IntegerTuple) GetData(index int) Feature {
   return this.Data[index];
}

func (this IntegerTuple) SetData(index int, newValue interface{}) {
   this.Data[index] = InferIntFeature(newValue);
}

func (this IntegerTuple) GetNumericData(index int) float64 {
   return this.Data[index].NumericValue();
}

func (this IntegerTuple) GetIntData(index int) int {
   return this.Data[index].IntValue();
}

func (this IntegerTuple) GetClass() Feature {
   return this.Class;
}

func (this IntegerTuple) DataSize() int {
   return len(this.Data);
}

// Util

func TupleEquals(a Tuple, b Tuple) bool {
   return reflect.DeepEqual(a, b);
}
