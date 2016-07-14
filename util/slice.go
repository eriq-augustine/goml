package util

import (
   "math/rand"
   "reflect"
)

// http://stackoverflow.com/a/12754757
func InterfaceSlice(slice interface{}) []interface{} {
   s := reflect.ValueOf(slice);
   if (s.Kind() != reflect.Slice) {
      panic("InterfaceSlice() given a non-slice type");
   }

   ret := make([]interface{}, s.Len());
   for i := 0; i < s.Len(); i++ {
      ret[i] = s.Index(i).Interface();
   }

   return ret;
}

// Fisher–Yates (Sattolo variant).
func ShuffleSlice(slice []interface{}) {
   for i, _ := range(slice) {
      var j int = rand.Intn(i + 1);
      slice[i], slice[j] = slice[j], slice[i];
   }
}
