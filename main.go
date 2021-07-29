package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/model3d/model2d"
)

func main() {
	for _, name := range []string{"train", "test"} {
		var dataset mnist.DataSet
		if name == "train" {
			dataset = mnist.LoadTrainingDataSet()
		} else {
			dataset = mnist.LoadTestingDataSet()
		}
		os.Mkdir(name, 0755)
		essentials.ConcurrentMap(0, len(dataset.Samples), func(i int) {
			sample := dataset.Samples[i]
			name := fmt.Sprintf("%s/%05d", name, i)
			outPath := name + ".json"
			if _, err := os.Stat(outPath); err == nil {
				return
			}
			mesh := SampleToMesh(sample)
			beziers := MeshToBeziers(mesh)
			obj := map[string]interface{}{
				"label":   sample.Label,
				"beziers": beziers,
			}
			data, err := json.Marshal(obj)
			essentials.Must(err)
			essentials.Must(ioutil.WriteFile(outPath, data, 0644))
		})
	}
}

func MeshToBeziers(m *model2d.Mesh) [][]model2d.BezierCurve {
	fitter := &model2d.BezierFitter{
		Tolerance: 1e-5,
		L2Penalty: 1e-8,
		Momentum:  0.5,
	}
	curves := fitter.Fit(m)
	var res [][]model2d.BezierCurve
	for len(curves) > 0 {
		subset := []model2d.BezierCurve{curves[0]}
		for i, c := range curves[1:] {
			subset = append(subset, c)
			if c[3] == curves[0][0] {
				curves = curves[i+2:]
				break
			}
		}
		res = append(res, subset)
	}
	return res
}

func SampleToMesh(sample mnist.Sample) *model2d.Mesh {
	bmp := model2d.NewBitmap(28, 28)
	for i, x := range sample.Intensities {
		if x > 0.5 {
			bmp.Data[i] = true
		}
	}
	mesh := bmp.Mesh()
	for _, iters := range []int{5, 4, 3, 2, 1, 0} {
		m := mesh.SmoothSq(iters)
		if m.Manifold() {
			mesh = m
			break
		}
	}
	mesh = mesh.Subdivide(1)
	for _, iters := range []int{20, 10, 5, 2, 1, 0} {
		m := mesh.SmoothSq(iters)
		if m.Manifold() {
			return m
		}
	}
	return mesh
}
