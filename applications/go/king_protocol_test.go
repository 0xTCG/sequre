package king

import (
	"fmt"
	"github.com/BurntSushi/toml"
	"go.dedis.ch/onet/v3/log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

var pid, _ = strconv.Atoi(os.Getenv("PID"))

func TestKingProtocol(t *testing.T) {
	log.LLvl1("Test")

	/*
		var X HorizontallyPartitionned
		var t PublicFloat64
		var het LocalVectorFloat64

		for i:= 0; i < len(X); i++ {
			het[i] = CountOnes(X[i])
		}

		tprime := (2 - 4*t)*het
		one := newMatriceOnes(len(X), len(X[0]))
		kinship := tprime - (X * X * one - 2 * X * X.T() + 1.T() * X.T() x X.T())
		relatives := IsPositive(kinship)
	*/

	prot := InitializeKingProtocol(pid, "config/")

	matrices := prot.KingProtocol()

	localInput := prot.Data

	// TODO loop in prot.ComparisonMap and if pid is in the list, then get the data
	for i, v := range prot.ComparisonMap {
		if i == strconv.Itoa(pid) {

			f_obt, err := os.Create("temp_king_go_obt_output.txt")
			if err != nil {
				log.Fatal(err)
			}
			defer f_obt.Close()

			f_exp, err := os.Create("temp_king_go_exp_output.txt")
			if err != nil {
				log.Fatal(err)
			}
			defer f_exp.Close()
		
			log.LLvl1("Test results of comparison with party", v)
			otherPartyConfig := new(ConfigKingProtocol)
			if _, err := toml.DecodeFile(filepath.Join("config/", fmt.Sprintf("configLocal.Party%d.toml", v[0])), otherPartyConfig); err != nil {
				fmt.Println(err)
			}
			readData, err := LoadDataset(otherPartyConfig.DataPath, []rune(otherPartyConfig.Separator)[0], false)
			if err != nil {
				log.Fatal(err)
			}
			// for now we use one example dataset, parties can read a subset of it
			otherData := readData[otherPartyConfig.StartingIndex : otherPartyConfig.StartingIndex+otherPartyConfig.NumberOfRows]
			for j := 0; j < otherPartyConfig.NumberOfRows; j++ {
				otherData[j] = otherData[j][:otherPartyConfig.NumberOfColumns]
			}
			log.LLvl1("otherData data:", len(otherData), len(otherData[0]))
			log.LLvl1("localInput data:", len(localInput), len(localInput[0]))
			for l := 0; l < len(localInput); l++ {
				for o := 0; o < len(otherData); o++ {
					minhet, localHet, otherHet := minNumberOfOnes(localInput[l], otherData[o])
					log.LLvl1("minhet:", minhet, localHet, otherHet)
					exp_value := squaredNormDistance(localInput[l], otherData[o]) / float64(minhet)
					obt_value := matrices[v[0]][o][l]
					
					_, err := fmt.Fprintln(f_exp, strconv.FormatFloat(exp_value, 'f', -1, 64))
					if err != nil {
						log.Fatal(err)
					}
					_, err = fmt.Fprintln(f_obt, strconv.FormatFloat(obt_value, 'f', -1, 64))
					if err != nil {
						log.Fatal(err)
					}
					
					log.LLvl1(l, o, "exp_value:", exp_value)
					log.LLvl1(l, o, "obt_value:", obt_value)
					log.LLvl1("Expected value:", exp_value, "Obtained value:", obt_value)
					if math.Abs(exp_value-obt_value) > 0.1 {
						log.Error("ERROR : ", l, o, localInput[l], otherData[o], "Expected value:", exp_value, "Obtained value:", obt_value)
					}
				}
			}

		}
	}

}

func squaredNormDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("Slices must have the same length")
	}

	var sumSquaredDiff float64
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sumSquaredDiff += diff * diff
	}

	return sumSquaredDiff
}

func minNumberOfOnes(a, b []float64) (int, int, int) {
	if len(a) != len(b) {
		panic("Slices must have the same length")
	}

	countOnesA := 0
	countOnesB := 0
	for i := 0; i < len(a); i++ {

		if a[i] == 1 {
			countOnesA++
		}
		if b[i] == 1 {
			countOnesB++
		}
	}

	minOnes := countOnesA
	if countOnesB < countOnesA {
		minOnes = countOnesB
	}

	return minOnes, countOnesA, countOnesB
}
