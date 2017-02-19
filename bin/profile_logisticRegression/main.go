package main

import (
   "os"
   "runtime/pprof"

   "github.com/eriq-augustine/goml/base"
   "github.com/eriq-augustine/goml/classification"
)

const (
   LR_CPU_PROFILE_FILENAME = "profile_logisticRegression.prof"
   LR_MEM_PROFILE_FILENAME = "profile_logisticRegression.mprof"
)

func main() {
   fakeTrainData := base.FakeData(2000, 3, 100, 0, nil, nil, 4);
   fakeTestData := base.FakeData(200, 3, 100, 0, nil, nil, 4);

   var cpuProfileOutPath string = LR_CPU_PROFILE_FILENAME;
   if (len(os.Args) > 1) {
      cpuProfileOutPath = os.Args[1];
   }

   cpuOutFile, err := os.Create(cpuProfileOutPath);
   if (err != nil) {
      panic("Could not create cpu profile file: " + err.Error());
   }
   defer cpuOutFile.Close();

   var memProfileOutPath string = LR_MEM_PROFILE_FILENAME;
   if (len(os.Args) > 2) {
      memProfileOutPath = os.Args[2];
   }

   memOutFile, err := os.Create(memProfileOutPath);
   if (err != nil) {
      panic("Could not create mem profile file: " + err.Error());
   }
   defer memOutFile.Close();

   pprof.StartCPUProfile(cpuOutFile);
   defer pprof.StopCPUProfile();

   var lr classification.Classifier = classification.NewLogisticRegression(nil, nil, -1);
   lr.Train(fakeTrainData);

   pprof.WriteHeapProfile(memOutFile);

   lr.Classify(fakeTestData);
}
