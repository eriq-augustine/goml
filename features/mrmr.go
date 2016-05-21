package features

import (
   "fmt"

   "github.com/eriq-augustine/goml/base"
)

const (
   NUM_FEATURES = 100
   NUM_BUCKETS = 100
)

// http://dl.acm.org/citation.cfm?id=1070809
type MRMRReducer struct{}

// TODO(eriq): Don't do anything if too few features.
func (reducer MRMRReducer) Init(data []base.Tuple) {
   if (len(data) == 0) {
      panic("Empty training set");
   }

   // TEST
   fmt.Println(data);

   // Discretize
   var discreteData []base.IntTuple = base.DiscretizeNumericFeatures(data, NUM_BUCKETS);

   // Marginal Probability (each feature)
   var marginalProbabilities [][]float64 = calcAllMarginalProbabilities(discreteData);

   // TEST
   fmt.Println(marginalProbabilities);

   // TODO(eriq);
   // Join Probability (each feature vs (each feature + class)
   // Calc
}

func (reducer MRMRReducer) Reduce(tuple base.Tuple) base.Tuple {
   // TODO(eriq)
   return tuple;
}

// Returns: [featureIndex][bucket (happens to be index]marginalProbibility
func calcAllMarginalProbabilities(discreteData []base.IntTuple) [][]float64 {
   var probabilities [][]float64 = make([][]float64, discreteData[0].DataSize());

   for featureIndex := 0; featureIndex < discreteData[0].DataSize(); featureIndex++ {
      probabilities[featureIndex] = calcFeatureMarginalProbabilities(discreteData, featureIndex);
   }

   return probabilities;
}

func calcFeatureMarginalProbabilities(discreteData []base.IntTuple, featureIndex int) []float64 {
   // Note that the data is already zero'd.
   var probabilities []float64 = make([]float64, NUM_BUCKETS);

   for _, tuple := range(discreteData) {
      probabilities[tuple.GetIntData(featureIndex)]++;
   }

   for i, _ := range(probabilities) {
      probabilities[i] /= float64(len(discreteData));
   }

   return probabilities;
}
