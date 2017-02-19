package classification

import (
   "testing"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/features"
   "github.com/eriq-augustine/goml/optimize"
)

type lrTestCase struct {
   Name string
   Reducer features.Reducer
   Optimizer optimize.Optimizer
   L2Penalty float64
   TestData []base.Tuple
   Input []base.Tuple
   ExpectedClasses []base.Feature
   MinExpectedConfidences []float64
}

func TestLogisticRegressionBase(t *testing.T) {
   fakeDefaultDataTest, fakeDefaultDataTestClasses := base.StripClasses([]base.Tuple(base.FakeDataDefault()));
   var fakeDefaultDataTestConfidences []float64 = make([]float64, len(fakeDefaultDataTest));
   for i, _ := range(fakeDefaultDataTest) {
      fakeDefaultDataTestConfidences[i] = 0.95;
   }

   fakeLargeDataTest, fakeLargeDataTestClasses := base.StripClasses([]base.Tuple(base.FakeData(200, 3, 100, 0, nil, nil, 4)));
   var fakeLargeDataTestConfidences []float64 = make([]float64, len(fakeLargeDataTest));
   for i, _ := range(fakeLargeDataTest) {
      fakeLargeDataTestConfidences[i] = 0.75;
   }

   var testCases []lrTestCase = []lrTestCase{
      lrTestCase{
         "Base - GD",
         features.NoReducer{},
         optimize.NewGradientDescent(0, 0, 0),
         1.0,
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 10}, 1),
            base.NewIntTuple([]interface{}{9, 9}, 1),
            base.NewIntTuple([]interface{}{11, 11}, 1),
            base.NewIntTuple([]interface{}{-10, -10}, 0),
            base.NewIntTuple([]interface{}{-9, -9}, 0),
            base.NewIntTuple([]interface{}{-11, -11}, 0),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{8, 8}, nil),
            base.NewIntTuple([]interface{}{-8, -8}, nil),
         },
         []base.Feature{
            base.Int(1),
            base.Int(0),
         },
         []float64{
            0.9,
            0.9,
         },
      },
      lrTestCase{
         "Base - SGD",
         features.NoReducer{},
         optimize.NewSGD(0, 0, 0, 0),
         1.0,
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 10}, 1),
            base.NewIntTuple([]interface{}{9, 9}, 1),
            base.NewIntTuple([]interface{}{11, 11}, 1),
            base.NewIntTuple([]interface{}{-10, -10}, 0),
            base.NewIntTuple([]interface{}{-9, -9}, 0),
            base.NewIntTuple([]interface{}{-11, -11}, 0),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{8, 8}, nil),
            base.NewIntTuple([]interface{}{-8, -8}, nil),
         },
         []base.Feature{
            base.Int(1),
            base.Int(0),
         },
         []float64{
            0.9,
            0.9,
         },
      },
      lrTestCase{
         "Defaults",
         nil,
         nil,
         -1,
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 10}, 1),
            base.NewIntTuple([]interface{}{9, 9}, 1),
            base.NewIntTuple([]interface{}{11, 11}, 1),
            base.NewIntTuple([]interface{}{-10, -10}, 0),
            base.NewIntTuple([]interface{}{-9, -9}, 0),
            base.NewIntTuple([]interface{}{-11, -11}, 0),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{8, 8}, nil),
            base.NewIntTuple([]interface{}{-8, -8}, nil),
         },
         []base.Feature{
            base.Int(1),
            base.Int(0),
         },
         []float64{
            0.9,
            0.9,
         },
      },
      lrTestCase{
         "Reduced",
         features.NewManualReducer([]int{1, 3}),
         nil,
         -1,
         []base.Tuple{
            base.NewIntTuple([]interface{}{1,  10, 0,  10, 6}, 1),
            base.NewIntTuple([]interface{}{2,  9,  0,  9,  5}, 1),
            base.NewIntTuple([]interface{}{3,  11, 0,  11, 4}, 1),
            base.NewIntTuple([]interface{}{4, -10, 0, -10, 3}, 0),
            base.NewIntTuple([]interface{}{5, -9,  0, -9,  2}, 0),
            base.NewIntTuple([]interface{}{6, -11, 0, -11, 1}, 0),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{1,  8, 0,  8, 2}, nil),
            base.NewIntTuple([]interface{}{2, -8, 0, -8, 1}, nil),
         },
         []base.Feature{
            base.Int(1),
            base.Int(0),
         },
         []float64{
            0.9,
            0.9,
         },
      },
      lrTestCase{
         "Confidence - 1",
         features.NoReducer{},
         nil,
         -1,
         []base.Tuple{
            base.NewIntTuple([]interface{}{1, 0}, 1),
            base.NewIntTuple([]interface{}{0, 1}, 1),
            base.NewIntTuple([]interface{}{-1, 0}, 1),
            base.NewIntTuple([]interface{}{-100, -100}, 0),
            base.NewIntTuple([]interface{}{-900, -900}, 0),
            base.NewIntTuple([]interface{}{-110, -110}, 0),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{0, 0}, nil),
         },
         []base.Feature{
            base.Int(1),
         },
         []float64{
            0.5,
         },
      },
      lrTestCase{
         "Confidence - 2",
         features.NoReducer{},
         nil,
         -1,
         []base.Tuple{
            base.NewIntTuple([]interface{}{10, 0}, 1),
            base.NewIntTuple([]interface{}{0, 10}, 1),
            base.NewIntTuple([]interface{}{-10, 0}, 1),
            base.NewIntTuple([]interface{}{-100, -100}, 0),
            base.NewIntTuple([]interface{}{-900, -900}, 0),
            base.NewIntTuple([]interface{}{-110, -110}, 0),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{0, 0}, nil),
         },
         []base.Feature{
            base.Int(1),
         },
         []float64{
            0.5,
         },
      },
      lrTestCase{
         "Three Class",
         features.NoReducer{},
         nil,
         -1,
         []base.Tuple{
            base.NewNumericTuple([]interface{}{1.7, 9.8}, 1),
            base.NewNumericTuple([]interface{}{1.2, 9.5}, 1),
            base.NewNumericTuple([]interface{}{1.5, 9.0}, 1),
            base.NewNumericTuple([]interface{}{1.6, 9.5}, 1),
            base.NewNumericTuple([]interface{}{1.5, 9.0}, 1),
            base.NewNumericTuple([]interface{}{5.1, 5.3}, 2),
            base.NewNumericTuple([]interface{}{5.2, 5.5}, 2),
            base.NewNumericTuple([]interface{}{5.5, 5.3}, 2),
            base.NewNumericTuple([]interface{}{5.2, 5.8}, 2),
            base.NewNumericTuple([]interface{}{5.0, 5.7}, 2),
            base.NewNumericTuple([]interface{}{9.7, 0.2}, 3),
            base.NewNumericTuple([]interface{}{9.8, 0.3}, 3),
            base.NewNumericTuple([]interface{}{9.2, 0.1}, 3),
            base.NewNumericTuple([]interface{}{9.5, 0.8}, 3),
            base.NewNumericTuple([]interface{}{9.9, 0.2}, 3),
         },
         []base.Tuple{
            base.NewNumericTuple([]interface{}{1.0, 10.0}, nil),
            base.NewNumericTuple([]interface{}{5.2, 5.5}, nil),
            base.NewNumericTuple([]interface{}{9.0, 0.0}, nil),
         },
         []base.Feature{
            base.Int(1),
            base.Int(2),
            base.Int(3),
         },
         []float64{
            0.9,
            0.75,
            0.9,
         },
      },
      lrTestCase{
         "Toy",
         features.NoReducer{},
         nil,
         -1,
         []base.Tuple{
            base.NewNumericTuple([]interface{}{6.02494983, 10.87741633}, 1),
            base.NewNumericTuple([]interface{}{-1.41197734, 6.019718083}, 1),
            base.NewNumericTuple([]interface{}{6.497017346, 7.506085862}, 1),
            base.NewNumericTuple([]interface{}{1.929010919, 2.580353403}, 0),
            base.NewNumericTuple([]interface{}{-5.263202539, 8.5527217}, 1),
            base.NewNumericTuple([]interface{}{-4.493846321, 6.788836776}, 1),
            base.NewNumericTuple([]interface{}{0.3433043254, 7.3929728}, 1),
            base.NewNumericTuple([]interface{}{1.733925363, 6.711393021}, 1),
            base.NewNumericTuple([]interface{}{6.993333946, 7.888648457}, 1),
            base.NewNumericTuple([]interface{}{3.884096905, 2.432574454}, 0),
            base.NewNumericTuple([]interface{}{8.884775104, 2.09224698}, 0),
            base.NewNumericTuple([]interface{}{-3.883163344, 2.443970517}, 0),
            base.NewNumericTuple([]interface{}{1.365159868, 2.696060741}, 0),
            base.NewNumericTuple([]interface{}{5.454786345, 9.139706521}, 1),
            base.NewNumericTuple([]interface{}{9.867796274, 2.293445021}, 0),
            base.NewNumericTuple([]interface{}{-3.263045263, 3.978708241}, 1),
            base.NewNumericTuple([]interface{}{4.601394389, 8.556057157}, 1),
            base.NewNumericTuple([]interface{}{8.748317183, 7.166432559}, 1),
            base.NewNumericTuple([]interface{}{2.138033305, 2.053114453}, 0),
            base.NewNumericTuple([]interface{}{-6.599880721, 2.721379511}, 0),
            base.NewNumericTuple([]interface{}{7.146121632, 2.968871929}, 0),
            base.NewNumericTuple([]interface{}{1.924721077, 2.99630803}, 0),
            base.NewNumericTuple([]interface{}{-5.500112054, 2.621530961}, 1),
            base.NewNumericTuple([]interface{}{3.418337738, 4.902474618}, 1),
            base.NewNumericTuple([]interface{}{-7.529701622, 8.900658982}, 1),
            base.NewNumericTuple([]interface{}{-4.127350297, 6.010394206}, 1),
            base.NewNumericTuple([]interface{}{9.193409243, 2.37459834}, 0),
            base.NewNumericTuple([]interface{}{-8.555467353, 7.632991542}, 1),
            base.NewNumericTuple([]interface{}{0.7083350283, 5.595037089}, 1),
            base.NewNumericTuple([]interface{}{-7.840612709, 4.600466688}, 1),
            base.NewNumericTuple([]interface{}{-3.860820574, 2.238430579}, 0),
            base.NewNumericTuple([]interface{}{-6.799366445, 2.887126603}, 0),
            base.NewNumericTuple([]interface{}{7.40927409, 5.364183826}, 1),
            base.NewNumericTuple([]interface{}{4.552144103, 9.239034339}, 1),
            base.NewNumericTuple([]interface{}{-0.2258836469, 4.354589373}, 1),
            base.NewNumericTuple([]interface{}{3.981023549, 5.997134104}, 1),
            base.NewNumericTuple([]interface{}{-6.509288989, 3.529688447}, 1),
            base.NewNumericTuple([]interface{}{-4.248242781, 4.339620677}, 1),
            base.NewNumericTuple([]interface{}{1.492180003, 7.31285105}, 1),
            base.NewNumericTuple([]interface{}{1.554444045, 5.952864031}, 1),
            base.NewNumericTuple([]interface{}{-0.6150883755, 3.282320366}, 1),
            base.NewNumericTuple([]interface{}{-3.005252107, 8.647368378}, 1),
            base.NewNumericTuple([]interface{}{3.677381689, 7.321482205}, 1),
            base.NewNumericTuple([]interface{}{-3.496313208, 5.767044115}, 1),
            base.NewNumericTuple([]interface{}{2.615699786, 2.793258365}, 0),
            base.NewNumericTuple([]interface{}{-7.325068618, 2.703827024}, 0),
            base.NewNumericTuple([]interface{}{1.768666677, 4.056616588}, 1),
            base.NewNumericTuple([]interface{}{3.123359235, 2.350084984}, 0),
            base.NewNumericTuple([]interface{}{5.909835779, 2.411465507}, 0),
            base.NewNumericTuple([]interface{}{-2.432473, 2.332377216}, 0),
            base.NewNumericTuple([]interface{}{9.039413928, 2.795425323}, 0),
            base.NewNumericTuple([]interface{}{-3.136192888, 4.296071735}, 1),
            base.NewNumericTuple([]interface{}{6.063545411, 2.618646446}, 0),
            base.NewNumericTuple([]interface{}{0.2702213225, 2.79234905}, 0),
            base.NewNumericTuple([]interface{}{-0.9764947404, 2.907119444}, 1),
            base.NewNumericTuple([]interface{}{1.145288243, 2.672468803}, 0),
            base.NewNumericTuple([]interface{}{7.185494521, 6.071816161}, 1),
            base.NewNumericTuple([]interface{}{-7.251282941, 7.733468362}, 1),
            base.NewNumericTuple([]interface{}{-2.55119898, 7.176663206}, 1),
            base.NewNumericTuple([]interface{}{9.377660903, 5.689898575}, 1),
            base.NewNumericTuple([]interface{}{9.387787377, 2.624445937}, 0),
            base.NewNumericTuple([]interface{}{-7.107130004, 2.552533871}, 0),
            base.NewNumericTuple([]interface{}{-5.731118958, 2.274801411}, 0),
            base.NewNumericTuple([]interface{}{5.509561923, 2.373773797}, 0),
            base.NewNumericTuple([]interface{}{9.076585951, 2.416160736}, 0),
            base.NewNumericTuple([]interface{}{-1.542551118, 6.006041253}, 1),
            base.NewNumericTuple([]interface{}{7.574549428, 2.771568722}, 0),
            base.NewNumericTuple([]interface{}{-2.798350928, 2.706602986}, 0),
            base.NewNumericTuple([]interface{}{3.141718117, 2.219109694}, 0),
            base.NewNumericTuple([]interface{}{-1.276559558, 2.962853423}, 0),
            base.NewNumericTuple([]interface{}{-2.994837774, 2.838692169}, 0),
            base.NewNumericTuple([]interface{}{-6.988421867, 2.345150554}, 0),
            base.NewNumericTuple([]interface{}{6.849948989, 7.625555325}, 1),
            base.NewNumericTuple([]interface{}{-2.688395818, 2.673888747}, 0),
            base.NewNumericTuple([]interface{}{1.169702717, 2.925238301}, 0),
            base.NewNumericTuple([]interface{}{-2.859187761, 2.089316478}, 0),
            base.NewNumericTuple([]interface{}{7.308618943, 2.080165724}, 0),
            base.NewNumericTuple([]interface{}{-8.719129518, 2.348989}, 0),
            base.NewNumericTuple([]interface{}{-8.338434254, 2.441078733}, 0),
            base.NewNumericTuple([]interface{}{6.479981456, 2.506214253}, 0),
            base.NewNumericTuple([]interface{}{-5.13198487, 5.318254938}, 1),
            base.NewNumericTuple([]interface{}{-8.536244831, 2.45545449}, 0),
            base.NewNumericTuple([]interface{}{9.491253637, 2.97676084}, 0),
            base.NewNumericTuple([]interface{}{4.670213381, 2.615947874}, 0),
            base.NewNumericTuple([]interface{}{-6.116855136, 2.276710529}, 0),
            base.NewNumericTuple([]interface{}{-5.222648227, 2.946679086}, 0),
            base.NewNumericTuple([]interface{}{4.692986169, 6.380182285}, 1),
            base.NewNumericTuple([]interface{}{3.61768358, 8.11353796}, 1),
            base.NewNumericTuple([]interface{}{-6.035780332, 2.353362232}, 0),
            base.NewNumericTuple([]interface{}{1.881069655, 3.229909632}, 1),
            base.NewNumericTuple([]interface{}{-8.546638454, 2.664028526}, 0),
            base.NewNumericTuple([]interface{}{4.205663878, 6.695923232}, 1),
            base.NewNumericTuple([]interface{}{6.551554062, 3.97164302}, 1),
            base.NewNumericTuple([]interface{}{9.884113811, 6.626539054}, 1),
            base.NewNumericTuple([]interface{}{7.150396708, 6.913478689}, 1),
            base.NewNumericTuple([]interface{}{-0.3489659823, 2.760042867}, 0),
            base.NewNumericTuple([]interface{}{7.374988586, 5.982823522}, 1),
            base.NewNumericTuple([]interface{}{-6.471857312, 2.026714791}, 0),
            base.NewNumericTuple([]interface{}{5.939700906, 2.951810542}, 0),
            base.NewNumericTuple([]interface{}{5.231824685, 7.192093605}, 1),
         },
         []base.Tuple{
            base.NewIntTuple([]interface{}{0, 8}, nil),
         },
         []base.Feature{
            base.Int(1),
         },
         []float64{
            0.9,
         },
      },
      lrTestCase{
         "FakeDataDefault",
         features.NoReducer{},
         nil,
         -1,
         base.FakeDataDefault(),
         fakeDefaultDataTest,
         fakeDefaultDataTestClasses,
         fakeDefaultDataTestConfidences,
      },
      lrTestCase{
         "FakeDataLarge",
         features.NoReducer{},
         nil,
         -1,
         base.FakeData(2000, 3, 100, 0, nil, nil, 4),
         fakeLargeDataTest,
         fakeLargeDataTestClasses,
         fakeLargeDataTestConfidences,
      },
   };

   for _, testCase := range(testCases) {
      var lr Classifier = NewLogisticRegression(testCase.Reducer, testCase.Optimizer, testCase.L2Penalty);
      lr.Train(testCase.TestData);
      var actualClasses []base.Feature;
      var actualConfidences []float64;

      actualClasses, actualConfidences = lr.Classify(testCase.Input);

      if (len(actualClasses) != len(testCase.ExpectedClasses)) {
         t.Errorf("(%s) -- Length of expected (%d) and actual classes (%d) do not match", testCase.Name, len(testCase.ExpectedClasses), len(actualClasses));
         continue;
      }

      if (len(actualConfidences) != len(testCase.MinExpectedConfidences)) {
         t.Errorf("(%s) -- Length of expected (%d) and actual confidences (%d) do not match", testCase.Name, len(testCase.MinExpectedConfidences), len(actualConfidences));
         continue;
      }

      // Go over each value explicitly to make output more readable.
      for i, _ := range(actualClasses) {
         if (actualClasses[i] != testCase.ExpectedClasses[i]) {
            t.Errorf("(%s)[%d] -- Bad classification. Expected class: %v, Got: %v", testCase.Name, i, testCase.ExpectedClasses[i], actualClasses[i]);
         }

         if (actualConfidences[i] < testCase.MinExpectedConfidences[i]) {
            t.Errorf("(%s)[%d] -- Bad classification. Expected confidence of at least: %v, Got: %v", testCase.Name, i, testCase.MinExpectedConfidences[i], actualConfidences[i]);
         }
      }
   }
}

func BenchmarkLogisticRegressionBase(b *testing.B) {
   fakeLargeDataTest, fakeLargeDataTestClasses := base.StripClasses([]base.Tuple(base.FakeData(200, 3, 100, 0, nil, nil, 4)));
   var fakeLargeDataTestConfidences []float64 = make([]float64, len(fakeLargeDataTest));
   for i, _ := range(fakeLargeDataTest) {
      fakeLargeDataTestConfidences[i] = 0.75;
   }

   var testCase lrTestCase = lrTestCase{
      "FakeDataLarge",
      features.NoReducer{},
      nil,
      -1,
      base.FakeData(2000, 3, 100, 0, nil, nil, 4),
      fakeLargeDataTest,
      fakeLargeDataTestClasses,
      fakeLargeDataTestConfidences,
   };

   // Ignore time setting up the data.
   b.ResetTimer();

   for n := 0; n < b.N; n++ {
      var lr Classifier = NewLogisticRegression(testCase.Reducer, testCase.Optimizer, testCase.L2Penalty);
      lr.Train(testCase.TestData);
      lr.Classify(testCase.Input);
   }
}
