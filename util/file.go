package util

import (
   "fmt"
   "math/rand"
   "path/filepath"
   "os"
)

const (
   RANDOM_NAME_LENGTH = 64
   RANDOM_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

// You should probably pass an extension, but the other two can be empty strings.
func TempFilePath(extension string, prefix string, suffix string) string {
   if (extension == "") {
      extension = "temp";
   }

   filename := prefix + RandomString(RANDOM_NAME_LENGTH) + suffix + "." + extension;
   return filepath.Join(os.TempDir(), filename);
}

func RandomString(length int) string {
   bytes := make([]byte, length);
   _, err := rand.Read(bytes);
   if (err != nil) {
      // TODO(eriq): Logs.
      // log.ErrorE("Unable to generate random string", err);
      panic(fmt.Sprintf("Unable to generate random string: %s", err));
   }

   for i, val := range(bytes) {
      bytes[i] = RANDOM_CHARS[int(val) % len(RANDOM_CHARS)];
   }

   return string(bytes)
}
