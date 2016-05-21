package util

import (
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
