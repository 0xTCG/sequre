package king

import (
	"encoding/csv"
	"errors"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"io"
	"os"
	"strconv"
)

func LoadDataset(path string, separator rune, header bool) ([][]float64, error) {

	data := TabulatedMatrixLoader{
		Path: path,
	}

	dataset, err := data.Load(separator, header)
	if err != nil {
		return nil, err
	}
	return dataset, nil
}

type TabulatedMatrixLoader struct{ Path string }

// Load loads the dataset at the given Path
func (c TabulatedMatrixLoader) Load(separator rune, header bool) ([][]float64, error) {
	ds, err := CSV{
		Path:      c.Path,
		Separator: separator,
		Header:    header,
	}.Load(false)

	return ds, err
}

// Loader loads a DataSet
// onlyInfo is to get the header of a DataSet, usually the number of column
type Loader interface {
	Load(onlyInfo bool) ([][]float64, error)
}

func LoadMatrix(loader Loader) (*mat.Dense, error) {
	dataset, err := loader.Load(false)

	if err != nil {
		return nil, err
	}

	return mat.NewDense(len(dataset), len(dataset[0]), Flatten(dataset)), nil
}

func Flatten(array [][]float64) []float64 {
	ret := make([]float64, len(array)*len(array[0]))
	lineLen := len(array[0])

	for i, col := range array {
		for j, el := range col {
			ret[i*lineLen+j] = el
		}
	}

	return ret
}

// Side is Left or Right
type Side bool

// CSV is a File splitting at the given Side
type CSV struct {
	Path      string
	Separator rune
	Split     Side
	Header    bool
}

var _ Loader = CSV{}

// Load loads the dataset found in the File
func (f CSV) Load(onlyInfo bool) ([][]float64, error) {
	matrix, err := f.loadMatrix(onlyInfo)
	if err != nil {
		return nil, fmt.Errorf("loading matrix: %v", err)
	}

	return matrix, err
}

func (f CSV) loadMatrix(onlyInfo bool) ([][]float64, error) {
	file, err := os.Open(f.Path)
	if err != nil {
		return nil, fmt.Errorf("opening file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = f.Separator
	reader.TrimLeadingSpace = true

	readHeader := false
	var records [][]float64
	for {
		record, err := reader.Read()
		if record == nil && errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading record: %w", err)
		}

		if f.Header && !readHeader {
			readHeader = true
			continue
		}

		line := make([]float64, len(record))
		for i, v := range record {
			parsed, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("parsing as float at line %v, element %v: %w", len(records), i, err)
			}
			line[i] = parsed
		}

		records = append(records, line)

		if onlyInfo {
			return records, nil
		}
	}

	return records, nil
}
