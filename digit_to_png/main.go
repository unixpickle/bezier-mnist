package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image/png"
	"io/ioutil"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model2d"
)

type FileObj struct {
	Label   int                     `json:"label"`
	Beziers [][]model2d.BezierCurve `json:"beziers"`
}

func main() {
	var resolution int
	var segmentsPerCurve int
	flag.IntVar(&resolution, "resolution", 256, "resolution to render at")
	flag.IntVar(&segmentsPerCurve, "segments", 100, "segments per curve")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: digit_to_png [flags] <input.json> <output.png>")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Flags:")
		flag.PrintDefaults()
		fmt.Fprintln(os.Stderr)
	}
	flag.Parse()

	args := flag.Args()
	if len(args) != 2 {
		flag.Usage()
		os.Exit(1)
	}
	inPath, outPath := args[0], args[1]

	inData, err := ioutil.ReadFile(inPath)
	essentials.Must(err)
	var obj FileObj
	essentials.Must(json.Unmarshal(inData, &obj))

	mesh := model2d.NewMesh()
	for _, loop := range obj.Beziers {
		for _, curve := range loop {
			mesh.AddMesh(model2d.CurveMesh(curve, segmentsPerCurve))
		}
	}
	collider := model2d.MeshToCollider(mesh)
	rend := &model2d.Rasterizer{
		// The resolution is always rounded up, so we
		// slightly shrink the scale.
		Scale:  (float64(resolution) / 28.0) * (1 - 1e-5),
		Bounds: model2d.NewRect(model2d.XY(0, 0), model2d.XY(28, 28)),
	}
	img := rend.RasterizeColliderSolid(collider)
	bounds := img.Bounds()
	if bounds.Dx() != resolution || bounds.Dy() != resolution {
		panic("resolution is incorrect")
	}

	// Invert the colors to imitate the standard MNIST style.
	for y := 0; y < resolution; y++ {
		for x := 0; x < resolution; x++ {
			color := img.GrayAt(x, y)
			color.Y = 0xff - color.Y
			img.SetGray(x, y, color)
		}
	}

	outFile, err := os.Create(outPath)
	essentials.Must(err)
	defer func() {
		essentials.Must(outFile.Close())
	}()
	essentials.Must(png.Encode(outFile, img))
}
