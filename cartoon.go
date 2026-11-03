package cartoon

import (
	"errors"
	"unsafe"

	"github.com/oxzjh/cv"
	"github.com/oxzjh/ort"
)

type Cartoon struct {
	engine *ort.Ort
	width  int
	height int
}

func (c *Cartoon) Extract(m *cv.Mat) (out *cv.Mat, err error) {
	inputBlob := m.BlobFromImage(1/127.5, cv.NewSize(c.width, c.height), cv.NewScalar(127.5, 127.5, 127.5, 0), true, true)
	defer inputBlob.Free()
	var result *ort.Result
	if result, err = c.engine.Infer(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), ort.TypeFloat); err != nil {
		return
	}
	defer result.Free()
	var pointer unsafe.Pointer
	if pointer, err = result.GetOutputPointer(0); err != nil {
		return
	}
	n := uintptr(c.width*c.height) << 2
	r := cv.NewMatWithData(c.height, c.width, cv.MatType32FC1, pointer)
	g := cv.NewMatWithData(c.height, c.width, cv.MatType32FC1, unsafe.Pointer(uintptr(pointer)+n))
	b := cv.NewMatWithData(c.height, c.width, cv.MatType32FC1, unsafe.Pointer(uintptr(pointer)+n<<1))
	output := cv.Merge([]cv.Mat{*b, *g, *r})
	r.Free()
	g.Free()
	b.Free()
	out = output.ConvertToAB(cv.MatType8UC3, 127.5, 127.5)
	output.Free()
	return
}

func (c *Cartoon) ExtractFile(file string) (*cv.Mat, error) {
	m := cv.IMRead(file, cv.FLAG_COLOR)
	defer m.Free()
	if m.Cols() < 4 || m.Rows() < 4 {
		return nil, errors.New("read file failed")
	}
	return c.Extract(m)
}

func (c *Cartoon) Close() {
	if c.engine != nil {
		c.engine.Close()
		c.engine = nil
	}
}

func New(model string, threads int) (*Cartoon, error) {
	engine, err := ort.New(model, "cartoon", threads)
	if err != nil {
		return nil, err
	}
	return &Cartoon{engine, int(engine.InputDims[0][3]), int(engine.InputDims[0][2])}, nil
}
