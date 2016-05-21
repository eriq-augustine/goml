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
func (reducer MRMRReducer) Init(features Features, data []base.Tuple) {
   if (len(data) == 0) {
      panic("Empty training set");
   }

   // TEST
   fmt.Println(len(features));
   fmt.Println(data);

   // Discretize
   var discreteData []base.Tuple = base.DiscretizeNumericFeatures(data, NUM_BUCKETS);

   // Marginal Probability (each feature)
   var marginalProbabilities [][]float64 = calcAllMarginalProbabilities(discreteData);

   // TEST
   fmt.Println(marginalProbabilities);

   // TODO(eriq);
   // Join Probability (each feature vs (each feature + class)
   // Calc
}

func (reducer MRMRReducer) Reduce(data []base.Tuple) []base.Tuple {
   // TODO(eriq)
   return data;
}

// Returns: [featureIndex][bucket (happens to be index]marginalProbibility
func calcAllMarginalProbabilities(discreteData []base.Tuple) [][]float64 {
   var probabilities [][]float64 = make([][]float64, len(discreteData[0].Data));

   for featureIndex := 0; featureIndex < len(discreteData[0].Data); featureIndex++ {
      probabilities[featureIndex] = calcFeatureMarginalProbabilities(discreteData, featureIndex);
   }

   return probabilities;
}

func calcFeatureMarginalProbabilities(discreteData []base.Tuple, featureIndex int) []float64 {
   // Note that the data is already zero'd.
   var probabilities []float64 = make([]float64, NUM_BUCKETS);

   for _, tuple := range(discreteData) {
      probabilities[tuple.Data[featureIndex].(int)]++;
   }

   for i, _ := range(probabilities) {
      probabilities[i] /= float64(len(discreteData));
   }

   return probabilities;
}
