package king

import (
	"go.dedis.ch/onet/v3/log"
	"os"
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

	prot.KingProtocol()

}
