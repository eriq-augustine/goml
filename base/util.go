package base

import (
   "math/rand"
   "reflect"
   "runtime"
   "time"

   "github.com/eriq-augustine/goml/util"
)

var random *rand.Rand = rand.New(rand.NewSource(time.Now().UnixNano()));
var maxProcs = runtime.GOMAXPROCS(0);
var machineMaxProcs = maxProcs;

func Seed(seed int64) {
   random = rand.New(rand.NewSource(seed));
}

func GetMaxProcs() int {
   return maxProcs;
}

// Set the maximum number of concurent goroutines a call to goml will use.
// Return the value that was actually used.
func SetMaxProcs(max int) int {
   maxProcs = util.MinInt(util.MaxInt(max, 0), machineMaxProcs);
   return maxProcs;
}

func TupleEquals(a Tuple, b Tuple) bool {
   return reflect.DeepEqual(a, b);
}

// Return a new tuple with the given type and data.
func NewTypedTuple(tupleType reflect.Type, data []interface{}, class interface{}) Tuple {
   var newTuple Tuple;

   if (tupleType == reflect.TypeOf(IntegerTuple{}) || tupleType == reflect.TypeOf(&IntegerTuple{})) {
      newTuple = NewIntTuple(data, class);
   } else if (tupleType == reflect.TypeOf(FloatTuple{}) || tupleType == reflect.TypeOf(&FloatTuple{})) {
      newTuple = NewNumericTuple(data, class);
   } else {
      newTuple = NewTuple(data, class);
   }

   return newTuple;
}

// Fisher–Yates (Sattolo variant).
func ShuffleTuples(slice []Tuple) {
   for i, _ := range(slice) {
      var j int = random.Intn(i + 1);
      slice[i], slice[j] = slice[j], slice[i];
   }
}

// Pull out tuples that match the given indexes.
func SelectTuples(tuples []Tuple, indexes []int) []Tuple {
   var rtn []Tuple = make([]Tuple, len(indexes));
   for i, chosenIndex := range(indexes) {
      rtn[i] = tuples[chosenIndex];
   }
   return rtn;
}

// Pull out tuples that match the given indexes.
func SelectNumericTuples(tuples []NumericTuple, indexes []int) []NumericTuple {
   var rtn []NumericTuple = make([]NumericTuple, len(indexes));
   for i, chosenIndex := range(indexes) {
      rtn[i] = tuples[chosenIndex];
   }
   return rtn;
}

// Shallow wrappers for util.FakeData().
// Wraps the output in NumericTuple.

func FakeDataDefault() []Tuple {
   data, classes := util.FakeDataDefault();

   var rtn []Tuple = make([]Tuple, len(data));
   for i, _ := range(data) {
      rtn[i] = NewFloatTuple(data[i], classes[i]);
   }

   return rtn;
}

func FakeData(numPoints int, numClasses int, numFeatures int, maxDistanceToCenter float64,
              rangeMin []float64, rangeMax []float64, seed int64) []Tuple {
   data, classes := util.FakeData(numPoints, numClasses, numFeatures, maxDistanceToCenter, rangeMin, rangeMax, seed);

   var rtn []Tuple = make([]Tuple, len(data));
   for i, _ := range(data) {
      rtn[i] = NewFloatTuple(data[i], classes[i]);
   }

   return rtn;
}

// Remove the classes from the tuples and return both the tuples and classes.
func StripClasses(tuples []Tuple) ([]Tuple, []Feature) {
   var rtnTuples []Tuple = make([]Tuple, len(tuples));
   var rtnClasses []Feature = make([]Feature, len(tuples));

   copy(rtnTuples, tuples);

   for i, _ := range(tuples) {
      rtnClasses[i] = rtnTuples[i].GetClass();
      rtnTuples[i].SetClass(nil);
   }

   return rtnTuples, rtnClasses;
}
