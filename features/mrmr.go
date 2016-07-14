package features

import (
   "fmt"
   "math"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/util"
)

const (
   DEFAULT_NUM_FEATURES = 100
   DEFAULT_NUM_BUCKETS = 20
)

// http://dl.acm.org/citation.cfm?id=1070809
type MRMRReducer struct{
   fanIn int
   fanOut int
   numFeatureBuckets int
   features []int
}

func NewMRMRReducer(numFeatures int, numBuckets int) *MRMRReducer {
   if (numFeatures <= 0) {
      numFeatures = DEFAULT_NUM_FEATURES;
   }

   if (numBuckets <= 0) {
      numBuckets = DEFAULT_NUM_BUCKETS;
   }

   return &MRMRReducer{
      fanIn: -1,
      fanOut: numFeatures,
      numFeatureBuckets: numBuckets,
   };
}

// It is required that all class labels be present in |data|.
func (this *MRMRReducer) Init(data []base.Tuple) {
   if (len(data) == 0) {
      panic("Empty training set");
   }

   this.fanIn = data[0].DataSize();
   if (this.fanIn <= this.fanOut) {
      // Done, great job!
      this.fanOut = this.fanIn;
      return;
   }

   // Discretize
   var discreteData []base.IntTuple = base.DiscretizeNumericFeatures(data, this.numFeatureBuckets);

   mutualInformation, classMutualInformation := this.calcAllMutualInformation(discreteData);

   // Calc
   this.features = this.chooseFeatures(this.fanIn, mutualInformation, classMutualInformation);
}

func (this MRMRReducer) Reduce(tuples []base.Tuple) []base.Tuple {
   return SelectFeatures(tuples, this.features);
}

func (this MRMRReducer) chooseFeatures(numFeatures int, mutualInformation [][]float64, classMutualInformation []float64) []int {
   var features []int = make([]int, this.fanOut);
   var usedFeatures map[int]bool = make(map[int]bool);

   for i := 0; i < this.fanOut; i++ {
      var bestFeatureIndex int = -1;
      var bestFeatureScore float64 = -1;

      for featureIndex := 0; featureIndex < numFeatures; featureIndex++ {
         _, hasFeature := usedFeatures[featureIndex];
         if (hasFeature) {
            continue;
         }

         var score float64 = this.calcMRMR(usedFeatures, featureIndex, mutualInformation, classMutualInformation);
         if (bestFeatureIndex == -1 || score > bestFeatureScore) {
            bestFeatureIndex = featureIndex;
            bestFeatureScore = score;
         }
      }

      features[i] = bestFeatureIndex;
      usedFeatures[bestFeatureIndex] = true;
   }

   return features;
}

func (this MRMRReducer) calcMRMR(usedFeatures map[int]bool, featureIndex int, mutualInformation [][]float64, classMutualInformation []float64) float64 {
   if (len(usedFeatures) == 0) {
      return classMutualInformation[featureIndex];
   }

   var sum float64;
   for usedFeatureIndex, _ := range(usedFeatures) {
      sum += mutualInformation[util.MaxInt(usedFeatureIndex, featureIndex)][util.MinInt(usedFeatureIndex, featureIndex)];
   }

   return classMutualInformation[featureIndex] - (sum / float64(len(usedFeatures)));
}

// Returns:
//    (non-class label mutual information, class label mutual information)
//    ([featureIndex1][featureIndex2] -> mutual information, [featureIndex] -> mutual information)
// Note that mutual information is symetric, so for the non-class variant,
// the same rules for calcAllJointProbabilities() will apply.
// (featureIndex1 > featureIndex2 and featureIndex1 == featureIndex2 is invalid).
func (this MRMRReducer) calcAllMutualInformation(discreteData []base.IntTuple) ([][]float64, []float64) {
   // Marginal Probability (each feature)
   var marginalProbabilities [][]float64 = this.calcAllMarginalProbabilities(discreteData);

   // Joint Probability (each feature vs (each feature + class)
   var jointProbabilities [][][][]float64 = this.calcAllJointProbabilities(discreteData);

   classMarginalProbabilities, classJointProbabilities := this.calcClassProbabilities(discreteData);

   // [featureIndex1][featureIndex2] -> mutual information
   var mutualInformation [][]float64 = make([][]float64, this.fanIn);
   for featureIndex1 := 0; featureIndex1 < this.fanIn; featureIndex1++ {
      mutualInformation[featureIndex1] = make([]float64, featureIndex1 + 1);

      // Invalid
      mutualInformation[featureIndex1][featureIndex1] = -1;

      for featureIndex2 := 0; featureIndex2 < featureIndex1; featureIndex2++ {
         mutualInformation[featureIndex1][featureIndex2] = this.calcMutualInformation(featureIndex1, featureIndex2, marginalProbabilities, jointProbabilities);
      }
   }

   // [featureIndex] -> mutual information
   var classMutualInformation []float64 = make([]float64, this.fanIn);
   for featureIndex := 0; featureIndex < this.fanIn; featureIndex++ {
      classMutualInformation[featureIndex] = this.calcMutualClassInformation(featureIndex, marginalProbabilities, classMarginalProbabilities, classJointProbabilities);
   }

   return mutualInformation, classMutualInformation;
}

func (this MRMRReducer) calcMutualInformation(featureIndex1 int, featureIndex2 int, marginalProbabilities [][]float64, jointProbabilities [][][][]float64) float64 {
   var mutualInfo float64;

   for featureBucket1 := 0; featureBucket1 < this.numFeatureBuckets; featureBucket1++ {
      for featureBucket2 := 0; featureBucket2 < this.numFeatureBuckets; featureBucket2++ {
         var jointProb float64 = jointProbabilities[featureIndex1][featureIndex2][featureBucket1][featureBucket2];
         var marginalProb1 = marginalProbabilities[featureIndex1][featureBucket1];
         var marginalProb2 = marginalProbabilities[featureIndex2][featureBucket2];

         if (jointProb == 0 || marginalProb1 == 0 || marginalProb2 == 0) {
            continue;
         }

         mutualInfo += jointProb * math.Log2(jointProb / (marginalProb1 * marginalProb2));
      }
   }

   return mutualInfo;
}

func (this MRMRReducer) calcMutualClassInformation(featureIndex int, marginalProbabilities [][]float64, classMarginalProbabilities []float64, classJointProbabilities [][][]float64) float64 {
   var mutualInfo float64;

   for classValueIndex := 0; classValueIndex < len(classJointProbabilities); classValueIndex++ {
      for featureBucket := 0; featureBucket < this.numFeatureBuckets; featureBucket++ {
         var jointProb float64 = classJointProbabilities[classValueIndex][featureIndex][featureBucket];
         var marginalProb1 = classMarginalProbabilities[classValueIndex];
         var marginalProb2 = marginalProbabilities[featureIndex][featureBucket];

         if (jointProb == 0 || marginalProb1 == 0 || marginalProb2 == 0) {
            continue;
         }

         mutualInfo += jointProb * math.Log2(jointProb / (marginalProb1 * marginalProb2));
      }
   }

   return mutualInfo;
}

// Calc marginal and joint probabilities for the class label.
func (this MRMRReducer) calcClassProbabilities(discreteData []base.IntTuple) ([]float64, [][][]float64) {
   // Map out the different class labels seen in the data and assign each an index.
   var classLabelIndex int = 0;
   var classValueMap map[base.Feature]int = make(map[base.Feature]int);

   for _, tuple := range(discreteData) {
      _, contains := classValueMap[tuple.GetClass()];
      if (!contains) {
         classValueMap[tuple.GetClass()] = classLabelIndex;
         classLabelIndex++;
      }
   }

   // Marginal probabilities
   // [classValueIndex] -> probability
   var marginalProbabilities []float64 = make([]float64, len(classValueMap));

   for _, tuple := range(discreteData) {
      marginalProbabilities[classValueMap[tuple.GetClass()]]++;
   }

   for i, _ := range(marginalProbabilities) {
      marginalProbabilities[i] /= float64(len(discreteData));
   }

   // Joint Probabilities
   // Initialize the entire solution space.
   // [classValueIndex][featureIndex][featureBucket] -> probability
   var jointProbabilities [][][]float64 = make([][][]float64, len(classValueMap));
   for _, classValueIndex := range(classValueMap) {
      jointProbabilities[classValueIndex] = make([][]float64, this.fanIn);
      for featureIndex, _ := range(jointProbabilities[classValueIndex]) {
         jointProbabilities[classValueIndex][featureIndex] = make([]float64, this.numFeatureBuckets);
      }
   }

   for featureIndex := 0; featureIndex < this.fanIn; featureIndex++ {
      for _, tuple := range(discreteData) {
         jointProbabilities[classValueMap[tuple.GetClass()]][featureIndex][tuple.GetIntData(featureIndex)]++;
      }
   }

   // Normalize
   for i, _ := range(jointProbabilities) {
      for j, _ := range(jointProbabilities[i]) {
         for k, _ := range(jointProbabilities[i][j]) {
            jointProbabilities[i][j][k] /= float64(len(discreteData));
         }
      }
   }

   return marginalProbabilities, jointProbabilities;
}

// Returns: [featureIndex][bucket (happens to be an index)]marginalProbibility
func (this MRMRReducer) calcAllMarginalProbabilities(discreteData []base.IntTuple) [][]float64 {
   var probabilities [][]float64 = make([][]float64, this.fanIn);

   for featureIndex := 0; featureIndex < this.fanIn; featureIndex++ {
      probabilities[featureIndex] = this.calcFeatureMarginalProbabilities(discreteData, featureIndex);
   }

   return probabilities;
}

func (this MRMRReducer) calcFeatureMarginalProbabilities(discreteData []base.IntTuple, featureIndex int) []float64 {
   // Note that the data is already zero'd.
   var probabilities []float64 = make([]float64, this.numFeatureBuckets);

   for _, tuple := range(discreteData) {
      probabilities[tuple.GetIntData(featureIndex)]++;
   }

   for i, _ := range(probabilities) {
      probabilities[i] /= float64(len(discreteData));
   }

   return probabilities;
}

// Returns: [featureIndex1][featureIndex2][feature1Bucket][feature2Bucket]
// Where featureIndex1 > featureIndex2 (we only need to calc half the solution space since it is symetric).
// Note that featureIndex1 < featureIndex2 is out of bounds and featureIndex1 == featureIndex2 is invalid.
// (We need the later to keep our indexes aligned.)
func (this MRMRReducer) calcAllJointProbabilities(discreteData []base.IntTuple) [][][][]float64 {
   var numFeatures int = this.fanIn;

   var probabilities [][][][]float64 = make([][][][]float64, numFeatures);

   for featureIndex1 := 0; featureIndex1 < numFeatures; featureIndex1++ {
      probabilities[featureIndex1] = make([][][]float64, featureIndex1 + 1);

      // Invalid.
      probabilities[featureIndex1][featureIndex1] = nil;

      for featureIndex2 := 0; featureIndex2 < featureIndex1; featureIndex2++ {
         probabilities[featureIndex1][featureIndex2] = this.calcFeatureJointProbabilities(discreteData, featureIndex1, featureIndex2);
      }
   }

   return probabilities;
}

func (this MRMRReducer) calcFeatureJointProbabilities(discreteData []base.IntTuple, featureIndex1 int, featureIndex2 int) [][]float64 {
   var probabilities [][]float64 = make([][]float64, this.numFeatureBuckets);
   for i := 0; i < this.numFeatureBuckets; i++ {
      probabilities[i] = make([]float64, this.numFeatureBuckets);
   }

   for _, tuple := range(discreteData) {
      probabilities[tuple.GetIntData(featureIndex1)][tuple.GetIntData(featureIndex2)]++;
   }

   for i, _ := range(probabilities) {
      for j, _ := range(probabilities[i]) {
         probabilities[i][j] /= float64(len(discreteData));
      }
   }

   return probabilities;
}

func (this MRMRReducer) printJointProbabilities(jointProbabilities [][][][]float64) {
   for featureIndex1, _ := range(jointProbabilities) {
      if (featureIndex1 == 0) {
         continue;
      }

      fmt.Println("Feature1: ", featureIndex1);

      for featureIndex2, _ := range(jointProbabilities[featureIndex1]) {
         if (featureIndex1 == featureIndex2) {
            continue;
         }

         fmt.Println("   Feature2: ", featureIndex2);

         // Header
         fmt.Print("      1\\2 |   ");
         for i := 0; i < len(jointProbabilities[featureIndex1][featureIndex2]); i++ {
            fmt.Printf("%03d    ", i);
         }
         fmt.Println();

         for feature1Bucket, _ := range(jointProbabilities[featureIndex1][featureIndex2]) {
            fmt.Printf("      %03d |", feature1Bucket);
            for feature2Bucket, _ := range(jointProbabilities[featureIndex1][featureIndex2][feature1Bucket]) {
               fmt.Printf(" %5.4f", jointProbabilities[featureIndex1][featureIndex2][feature1Bucket][feature2Bucket]);
            }
            fmt.Println();
         }
      }
   }
}
