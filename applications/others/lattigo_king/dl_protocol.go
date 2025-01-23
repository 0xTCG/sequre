// 188 of compressed LOC
// without minimum: 142
package king

import (
	"fmt"
	"github.com/BurntSushi/toml"
	"github.com/hhcho/sfgwas-private/cryptobasics"
	"github.com/hhcho/sfgwas-private/libspindle"
	"go.dedis.ch/onet/v3/log"
	"gonum.org/v1/gonum/mat"
	"os"
	"path/filepath"
	"time"
)

// MatMultInfo contains protocol specific infos and link to generic protocol structure
type DLProtocolInfo struct {
	Prot *ProtocolInfo
	Data [][]float64
}

// ConfigKingProtocol parameters to configure the protocol
type ConfigDLProtocol struct {
	DataPath        string `toml:"simple_data_path"`
	Separator       string `toml:"separator"`
	NumberOfRows    int    `toml:"number_of_rows"`
	NumberOfColumns int    `toml:"number_of_columns"`
	OutDir          string `toml:"output_dir"`
	CacheDir        string `toml:"cache_dir"`
}

// InitializeMatMultProtocol initialized the protocol parameters with global and local config files, and reads the input data
func InitializeDLProtocol(pid int, configFolder string) (matMultProt *DLProtocolInfo) {

	ConfigDLGlob := new(ConfigDLProtocol)
	if _, err := toml.DecodeFile(filepath.Join(configFolder, "configGlobal.toml"), ConfigDLGlob); err != nil {
		fmt.Println(err)
		return nil
	}

	ConfigDL := new(ConfigMatMultProtocol)
	if _, err := toml.DecodeFile(filepath.Join(configFolder, fmt.Sprintf("configLocal.Party%d.toml", pid)), ConfigDL); err != nil {
		fmt.Println(err)
		return nil
	}

	// Create cache/output directories
	if err := os.MkdirAll(ConfigDL.CacheDir, 0755); err != nil {
		panic(err)
	}
	if err := os.MkdirAll(ConfigDL.OutDir, 0755); err != nil {
		panic(err)
	}

	data := make([][]float64, ConfigDL.NumberOfRows)
	if pid > 0 {
		readData, err := LoadDataset(ConfigDL.DataPath, []rune(ConfigDL.Separator)[0], false)
		if err != nil {
			log.Fatal(err)
		}
		// for now we use one example dataset, parties can read a subset of it
		data = readData[0:ConfigDL.NumberOfRows]
		for j := 0; j < ConfigDL.NumberOfRows; j++ {
			data[j] = data[j][:ConfigDL.NumberOfColumns]
		}
		log.LLvl1(pid, " has data with dims:", len(data), len(data[0]))
	}

	prot := InitializeProtocol(pid, configFolder)

	return &DLProtocolInfo{
		Data: data,
		Prot: prot,
	}
}

// KingProtocol computes kinship coefficients among all paiwise values between matrices held by different parties
func (pi *DLProtocolInfo) DLProtocol() [][]float64 {
	log.LLvl1(time.Now(), "Finished Setup")

	pid := pi.Prot.MpcObj[0].GetPid()
	cps := pi.Prot.Cps

	log.LLvl1(time.Now(), "Start protocol")
	// exchange number of rows
	allPidsNbrRows := make([]int, pi.Prot.Config.NumMainParties+1)
	for otherPids := 0; otherPids < pi.Prot.Config.NumMainParties+1; otherPids++ {
		if otherPids == pid {
			if pid > 0 {
				allPidsNbrRows[otherPids] = len(pi.Data)
			}
		} else {
			if pid > 0 {
				log.LLvl1(pid, " send to ", otherPids)
				pi.Prot.MpcObj.GetNetworks()[otherPids].SendInt(len(pi.Data), otherPids)
			}
			if otherPids != 0 {
				log.LLvl1(pid, " receive from ", otherPids)
				allPidsNbrRows[otherPids] = pi.Prot.MpcObj.GetNetworks()[pid].ReceiveInt(otherPids)
			}
		}
	}
	log.LLvl1(pid, "finished exchanging number of rows: ", allPidsNbrRows)

	// for toy examples, we assume #SNPs >> #samples and therefore pack the input matrices row-wise (each row is one or multiple ciphertexts
	if pid > 0 {
		rowsLocal := len(pi.Data)
		cols := len(pi.Data[0])

		// all steps are local in cleartext except one addition required for each convolution
		inputVector := make([][]int, rowsLocal)
		count := 1
		if pid == 2 {
			count = count + rowsLocal // assume two parties with same number of rows
		}
		for i := range inputVector {
			inputVector[i] = make([]int, cols)
			for j := range inputVector[i] {
				inputVector[i][j] = count
				count++
			}
		}

		gx := [][]int{{1, 0}, {0, -1}}
		gy := [][]int{{0, 1}, {-1, 0}}

		gxRes1 := convolutionStep(inputVector, gx, rowsLocal, cols)
		log.LLvl1("gxRes1 : ", len(gxRes1), len(gxRes1[0]))
		gyRes1 := convolutionStep(inputVector, gy, rowsLocal, cols)

		result := make([][]int, len(gxRes1))
		for i := range gxRes1 {
			result[i] = make([]int, len(gxRes1[i]))
			for j := range gxRes1[i] {
				result[i][j] = 2*gxRes1[i][j]*gxRes1[i][j] + 3*gxRes1[i][j] + 7 + 2*gyRes1[i][j]*gyRes1[i][j] + 3*gyRes1[i][j] + 7
			}
		}
		log.LLvl1("result : ", len(result), len(result[0]))

		h_out, w_out := rowsLocal-1, cols-1
		matrix := make([][]int, h_out*w_out)
		count = 1
		for i := 0; i < h_out*w_out; i++ {
			matrix[i] = make([]int, w_out)
			for j := 0; j < w_out; j++ {
				matrix[i][j] = count
				count++
			}
		}
		log.LLvl1("matrix : ", len(matrix), len(matrix[0]))
		resultFinal := make([][]float64, w_out)
		for i := range gxRes1 {
			for j := range gxRes1[i] {
				for k := range matrix {
					//fmt.Println("asdfa", i, j)
					for l := range matrix[k] {
						if l == 0 {
							resultFinal[i] = make([]float64, w_out)
						}
							resultFinal[i][l] += float64(matrix[k][l] * gxRes1[i][j])

					}
				}
			}
		}
		//simulate missing part in convolution, due to horizontal split
		convolMissingLine := mat.NewDense(cols, 1, nil)
		convolMissingLineEnc := cryptobasics.EncryptDense(cps, mat.DenseCopyOf(convolMissingLine))
		squared := cryptobasics.CMult(cps, convolMissingLineEnc[0], convolMissingLineEnc[0])
		squared = cryptobasics.CMultConstMat(cps, libspindle.CipherMatrix{squared}, 2, false)[0]
		convolMissingLineEncTimes3 := cryptobasics.CMultConstMat(cps, libspindle.CipherMatrix{squared}, 3, false)[0]
		addAll := cryptobasics.CAdd(cps, squared, convolMissingLineEncTimes3)
		addAll = cryptobasics.CRescale(cps, addAll)
		convolMissingLineEncode := cryptobasics.EncodeDense(cps, mat.DenseCopyOf(convolMissingLine))
		cryptobasics.CPAdd(cps, addAll, convolMissingLineEncode[0])

		resultFinalDense := mat.NewDense(w_out, w_out, Flatten(resultFinal))
		resultFinalDenseEnc := cryptobasics.EncryptDense(cps, mat.DenseCopyOf(resultFinalDense.T()))
		resultFinalAggr := pi.Prot.MpcObj[0].Network.AggregateCMat(cps, resultFinalDenseEnc)
		resultDecrypt, _ := pi.Prot.MpcObj[0].Network.CollectiveDecryptMat(cps, resultFinalAggr, -1)
		decryptedMatrix := make([][]float64, len(resultDecrypt))
		for d := 0; d < len(resultDecrypt); d++ {
			decryptedMatrix[d] = libspindle.DecodeFloatVector(cps, resultDecrypt[d])
			//log.LLvl1("localResultDecode : ", decryptedMatrix[d][:2*allPidsNbrRows[otherPid]])
		}

	}
	return nil
}

func convolutionStep(input [][]int, kernel [][]int, h, w int) [][]int {
	hOut := h - len(kernel) + 1
	wOut := w - len(kernel[0]) + 1

	result := make([][]int, hOut)
	for i := range result {
		result[i] = make([]int, wOut)
		for j := range result[i] {
			sum := 0
			for ki := range kernel {
				for kj := range kernel[ki] {
					sum += input[i+ki][j+kj] * kernel[ki][kj]
				}
			}
			result[i][j] = sum
		}
	}
	return result
}
