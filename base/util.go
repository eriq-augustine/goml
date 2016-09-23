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

   if (tupleType == reflect.TypeOf(IntegerTuple{})) {
      newTuple = NewIntTuple(data, class);
   } else if (tupleType == reflect.TypeOf(FloatTuple{})) {
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
