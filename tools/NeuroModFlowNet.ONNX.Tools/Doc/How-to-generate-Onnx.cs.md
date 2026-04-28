1. Clone ONNX repository
	git clone https://github.com/onnx/onnx.git
1. Download protoc compiler Get the protoc-*.zip for your OS from the [Protobuf Releases](https://github.com/protocolbuffers/protobuf/releases).
   https://github.com/protocolbuffers/protobuf
1. Copy **protoc.exe** from the zip into the folder: onnx/onnx/ (where **onnx.proto** is located).
1. Run `protoc --csharp_out=. onnx.proto`
1. **Done!** You now have **Onnx.cs**. Add it to your project and enjoy.