package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/model3d/model2d"
)

const Version = 2

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
			var mesh *model2d.Mesh
			if Version == 1 {
				mesh = SampleToMesh(sample)
			} else {
				mesh = SampleToMeshV2(sample)
			}
			beziers := MeshToBeziers(mesh)

			obj := map[string]interface{}{
				"label":   sample.Label,
				"beziers": beziers,
			}
			data, err := json.Marshal(obj)
			essentials.Must(err)
			essentials.Must(ioutil.WriteFile(outPath, data, 0644))

			total := 0
			for _, loop := range beziers {
				total += len(loop)
			}
			log.Printf("Created sample %s (label %d): loops=%d, curves=%d",
				name, sample.Label, len(beziers), total)

			// Render the mesh we were trying to fit:
			// model2d.Rasterize(outPath+".png", mesh, 10.0)
		})
	}
}

func MeshToBeziers(m *model2d.Mesh) [][]model2d.BezierCurve {
	var res [][]model2d.BezierCurve
	for _, h := range model2d.MeshToHierarchy(m) {
		if h.Mesh.Area() < 5.0 {
			// Skip small blobs, since these are typically noise
			// and are perceptually unimportant.
			continue
		}
		res = append(res, HierarchyToBeziers(h)...)
	}
	return res
}

func HierarchyToBeziers(m *model2d.MeshHierarchy) [][]model2d.BezierCurve {
	segs := m.Mesh.SegmentSlice()
	if len(segs) == 0 {
		return nil
	}
	seg := segs[0]
	points := make([]model2d.Coord, 0, len(segs)+1)
	points = append(points, seg[0], seg[1])
	m.Mesh.Remove(seg)
	for i := 1; i < len(segs); i++ {
		next := m.Mesh.Find(seg[1])
		if len(next) != 1 {
			panic("mesh is non-manifold")
		}
		seg = next[0]
		m.Mesh.Remove(seg)
		points = append(points, seg[1])
	}

	numTries := 1
	if Version == 2 {
		numTries = 3
	}

	var curves []model2d.BezierCurve
	shortest := math.Inf(1)
	for i := 0; i < numTries; i++ {
		chain := FitChain(points)
		length := ChainLength(chain)
		if length < shortest {
			shortest = length
			curves = chain
		}
	}

	res := [][]model2d.BezierCurve{curves}
	for _, child := range m.Children {
		res = append(res, HierarchyToBeziers(child)...)
	}
	return res
}

func FitChain(points []model2d.Coord) []model2d.BezierCurve {
	fitter := &model2d.BezierFitter{
		Tolerance: 1e-5,
		L2Penalty: 1e-8,
		Momentum:  0.5,
	}
	if Version == 2 {
		// Settings to make search more precise (but expensive).
		fitter.NoFirstGuess = true
		fitter.LineStep = 1.5
		fitter.NumIters = 200
		fitter.Delta = 1e-7

		// Settings to reduce strange anomalies such as un-smooth
		// kinks, jagged small loops, and near-singularities.
		fitter.L2Penalty = 0
		fitter.AbsTolerance = 5e-6
		fitter.PerimPenalty = 1e-4
	}
	points = points[:len(points)-1]
	for {
		// Reshuffle the points to try different orderings
		// every time this method is called, or every time a
		// NaN/Inf is encountered.
		newStart := rand.Intn(len(points))
		points = append(append([]model2d.Coord{}, points[newStart:]...), points[:newStart]...)

		curves := fitter.FitChain(points, true)
		if ValidateChain(curves) {
			return curves
		}
		log.Println("Retrying after Bezier curves contained invalid values")
	}
}

func ValidateChain(c []model2d.BezierCurve) bool {
	for _, x := range c {
		for _, p := range x {
			s := p.Norm()
			if math.IsInf(s, 0) || math.IsNaN(s) {
				return false
			}
		}
	}
	return true
}

func ChainLength(c []model2d.BezierCurve) float64 {
	var res float64
	for _, curve := range c {
		res += curve.Length(1e-4, 0)
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

func SampleToMeshV2(sample mnist.Sample) *model2d.Mesh {
	var bestMesh *model2d.Mesh
	bestError := math.Inf(1)

	for thresh := 0.1; thresh < 0.9; thresh += 0.05 {
		mesh := SampleToMeshV2Threshold(sample, thresh)
		rast := model2d.Rasterizer{
			Bounds: model2d.NewRect(model2d.XY(0, 0), model2d.XY(28, 28)),
		}
		out := rast.RasterizeColliderSolid(model2d.MeshToCollider(mesh))
		var squaredError float64
		for y := 0; y < 28; y++ {
			for x := 0; x < 28; x++ {
				r, _, _, _ := out.At(x, y).RGBA()
				renderPixel := 1.0 - float64(r)/0xffff
				mnistPixel := sample.Intensities[x+y*28]
				squaredError += math.Pow(renderPixel-mnistPixel, 2)
			}
		}
		if squaredError < bestError {
			bestError = squaredError
			bestMesh = mesh
		}
	}
	return bestMesh
}

func SampleToMeshV2Threshold(sample mnist.Sample, threshold float64) *model2d.Mesh {
	img := image.NewGray(image.Rect(0, 0, 28, 28))
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			value := sample.Intensities[x+y*28]
			img.SetGray(x, y, color.Gray{Y: uint8(value * 0xff)})
		}
	}
	bmp := model2d.NewInterpBitmap(img, func(c color.Color) bool {
		r, _, _, _ := c.RGBA()
		return r > uint32(0xffff*threshold)
	})
	mesh := model2d.MarchingSquaresSearch(bmp, 0.5, 8)
	for _, iters := range []int{30, 20, 15, 10, 5, 4, 3, 2, 1, 0} {
		m := mesh.SmoothSq(iters)
		if m.Manifold() {
			mesh = m
			break
		}
	}
	// Create a new, non-intersecting mesh that is smoother and
	// hopefully undoes some of the shrinkage from SmoothSq().
	collider := model2d.MeshToCollider(mesh)
	solid := model2d.NewColliderSolidInset(collider, -0.1)
	mesh = model2d.MarchingSquaresSearch(solid, 0.25, 8)
	return mesh
}
