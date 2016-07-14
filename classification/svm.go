package classification

// For now, we will use a port of libsvm.
// Eventually, we would like to use our own implementation optimized for go.
// Currently, this only supports Numeric tuples with binary classes.

import (
   "bytes"
   "fmt"
   "os"

   "github.com/ewalker544/libsvm-go"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/features"
   "github.com/eriq-augustine/goml/util"
)

type Svm struct {
   reducer features.Reducer
   params *libSvm.Parameter
   model *libSvm.Model
   // Keep track of what kind of classes we are using so we can cast predictions accordingly.
   classFeatureType base.Feature
}

func NewSvm(reducer features.Reducer) *Svm {
   if (reducer == nil) {
      reducer = features.NoReducer{};
   }

   var svm Svm = Svm{
      reducer: reducer,
      params: libSvm.NewParameter(),
      model: nil,
      classFeatureType: nil,
   };

   return &svm;
}

func (this *Svm) Train(rawData []base.Tuple) {
   this.reducer.Init(rawData);
   rawData = this.reducer.Reduce(rawData);

   var data []base.NumericTuple = make([]base.NumericTuple, len(rawData));
   for i, tuple := range(rawData) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("SVM only supports taining on NumericTuple");
      }

      if (!numericTuple.GetClass().IsNumeric()) {
         panic("SVM only supports training on numeric classes");
      }

      data[i] = numericTuple;

      if (this.classFeatureType == nil) {
         this.classFeatureType = numericTuple.GetClass();
      }
   }

   this.model = libSvm.NewModel(this.params);

   // TODO(eriq): Training data not from file?
   var trainingFile string = writeTrainingFile(data);

   problem, err := libSvm.NewProblem(writeTrainingFile(data), this.params);
   if (err != nil) {
      panic(fmt.Sprintf("Failed to generate libsvm problem: %s", err));
   }

   this.model.Train(problem);
   os.Remove(trainingFile);
}

func (this Svm) Classify(tuples []base.Tuple) []base.Feature {
   tuples = this.reducer.Reduce(tuples);

   var results []base.Feature = make([]base.Feature, len(tuples));

   for i, tuple := range(tuples) {
      numericTuple, ok := tuple.(base.NumericTuple);
      if (!ok) {
         panic("SVM only supports classifying NumericTuple");
      }

      results[i] = this.classifySingle(numericTuple);
   }

   return results;
}

func (this Svm) classifySingle(numericTuple base.NumericTuple) base.Feature {
   var svmTuple map[int]float64 = tupleToSVMTuple(numericTuple);
   return this.castClassFeature(this.model.Predict(svmTuple));
}

func (this Svm) castClassFeature(val float64) base.Feature {
   switch featureType := this.classFeatureType.(type) {
   case base.IntFeature:
      return base.Int(int(val));
   case base.FloatFeature:
      return base.Float(val);
   case base.BoolFeature:
      return base.Bool(util.FloatToBool(val));
   default:
      panic(fmt.Sprintf("SVM only takes numeric class features, found: %T", featureType));
   }
}

func tupleToSVMTuple (numericTuple base.NumericTuple) map[int]float64 {
   var svmTuple map[int]float64 = make(map[int]float64);

   for i := 0; i < numericTuple.DataSize(); i++ {
      // libsvm 1-indexes features.
      svmTuple[i + 1] = numericTuple.GetNumericData(i);
   }

   return svmTuple;
}

// Write the training data to a file and return the path.
func writeTrainingFile(data []base.NumericTuple) string {
   var tempPath string = util.TempFilePath("train", "svm_", "");

   file, err := os.Create(tempPath);
   if (err != nil) {
      panic(fmt.Sprintf("Unable to create training file: %s", err));
   }
   defer file.Close();

   for _, tuple := range(data) {
      fmt.Fprintf(file, "%s\n", tupleToTrainingString(tuple));
   }

   return tempPath;
}

func tupleToTrainingString(tuple base.NumericTuple) string {
   var buf *bytes.Buffer = new(bytes.Buffer);

   // Class first.
   fmt.Fprintf(buf, "%v", util.NumericValue(tuple.GetClass().Value()));

   // Each feature.
   for i := 0; i < tuple.DataSize(); i++ {
      // libsvm 1-indexes features.
      fmt.Fprintf(buf, " %d:%v", i + 1, tuple.GetNumericData(i));
   }

   return buf.String();
}
