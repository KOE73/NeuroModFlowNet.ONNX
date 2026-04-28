# За что отвечает этот проект? (Например: "Обертки над FFMPEG и работа с памятью").

- Просмотр учтройства *.onnx файлов
- Добавление слоев для прямой загрузки Mat в сети
 

# От кого он зависит?
## Onnx.cs
Генерируется вручную 
1. Clone ONNX repository
	git clone https://github.com/onnx/onnx.git
1. Download protoc compiler Get the protoc-*.zip for your OS from the [Protobuf Releases](https://github.com/protocolbuffers/protobuf/releases).
   https://github.com/protocolbuffers/protobuf
1. Copy **protoc.exe** from the zip into the folder: onnx/onnx/ (where **onnx.proto** is located).
1. Run `protoc --csharp_out=. onnx.proto`
1. **Done!** You now have **Onnx.cs**. Add it to your project and enjoy.

# Кто от него зависит?
Особо никто не зависит, но можно использовать классы в своих инструментах.

# Почему это выделено в отдельную сборку?
- Особо в работе не надо никому, а вот для анализа *.onnx и донастройки их норм
- Иметь отдельную утилиту для автоматизированной донастройки сетей после обучения
- Встраивать в последовательность сборки может быть
