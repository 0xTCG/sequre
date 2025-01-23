package king

import (
	"go.dedis.ch/onet/v3/log"
	"testing"
	"time"
	"fmt"
)

func TestMatMultProtocol(t *testing.T) {
	log.LLvl1("Test")

	prot := InitializeMatMultProtocol(pid, "config/matmult")

	prot.Prot.MpcObj.GetNetworks().ResetNetworkLog()

	var start time.Time

	start = time.Now()
	matrix := prot.MatMultProtocol()
	fmt.Printf("Matmul done in %s \n", time.Since(start))

	if pid == 1 {
		log.LLvl1("Size of matrix :", len(matrix), len(matrix[0]))
		//log.LLvl1(matrix)
	}

	prot.Prot.MpcObj.GetNetworks().PrintNetworkLog()

}
