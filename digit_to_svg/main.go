package main

import (
	"encoding/json"
	"fmt"
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
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: digit_to_svg <input.json> <output.svg>")
		os.Exit(1)
	}
	inPath, outPath := os.Args[1], os.Args[2]
	inData, err := ioutil.ReadFile(inPath)
	essentials.Must(err)
	var obj FileObj
	essentials.Must(json.Unmarshal(inData, &obj))

	pathData := ""
	for _, loop := range obj.Beziers {
		loopData := fmt.Sprintf("M %f,%f ", loop[0][0].X, loop[0][0].Y)
		for _, curve := range loop {
			loopData += fmt.Sprintf("C %f,%f %f,%f %f,%f ", curve[1].X, curve[1].Y,
				curve[2].X, curve[2].Y, curve[3].X, curve[3].Y)
		}
		loopData += "Z"
		pathData += loopData
	}
	outData := []byte(`<?xml version="1.0" encoding="utf-8" ?>` +
		`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28">` +
		`<rect x="0" y="0" width="28" height="28" fill="black" />` +
		`<path fill="white" d="` + pathData + `" />` +
		`</svg>`)
	essentials.Must(ioutil.WriteFile(outPath, outData, 0644))
}
