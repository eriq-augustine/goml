package base

import (
	"math"
	"testing"

	"github.com/eriq-augustine/goml/util"
)

type DistanceTestData struct {
	Title    string
	A        Tuple
	B        Tuple
	Distance float64
}

func TestEuclidean(t *testing.T) {
	var distancer Distancer = Euclidean{}

	var testData []DistanceTestData = []DistanceTestData{
		DistanceTestData{
			"Zero Value",
			Tuple{[]interface{}{}, nil},
			Tuple{[]interface{}{}, nil},
			0,
		},
		DistanceTestData{
			"Single Value - Same",
			Tuple{[]interface{}{1}, nil},
			Tuple{[]interface{}{1}, nil},
			0,
		},
		DistanceTestData{
			"Single Value - Diff",
			Tuple{[]interface{}{2}, nil},
			Tuple{[]interface{}{10}, nil},
			8,
		},
		DistanceTestData{
			"Two Value - Same",
			Tuple{[]interface{}{4, 4}, nil},
			Tuple{[]interface{}{4, 4}, nil},
			0,
		},
		DistanceTestData{
			"Two Value - Diff",
			Tuple{[]interface{}{1, -1}, nil},
			Tuple{[]interface{}{-1, 1}, nil},
			math.Sqrt(8),
		},
		DistanceTestData{
			"Three Value - Same",
			Tuple{[]interface{}{1, 2, 3}, nil},
			Tuple{[]interface{}{1, 2, 3}, nil},
			0,
		},
		DistanceTestData{
			"Three Value - Diff",
			Tuple{[]interface{}{1, 2, 3}, nil},
			Tuple{[]interface{}{3, 2, 1}, nil},
			math.Sqrt(8),
		},
	}

	for _, testCase := range testData {
		var actual float64 = distancer.Distance(testCase.A, testCase.B)
		if !util.FloatEquals(actual, testCase.Distance) {
			t.Errorf("Euclidean distance error (%s). Expected: %v, Got: %v", testCase.Title, testCase.Distance, actual)
		}
	}
}
