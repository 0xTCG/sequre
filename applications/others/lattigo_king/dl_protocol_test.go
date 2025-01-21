package king

import (
	"go.dedis.ch/onet/v3/log"
	"testing"
)

func TestDLProtocol(t *testing.T) {
	log.LLvl1("Test")

	prot := InitializeDLProtocol(pid, "config/dl")

	prot.Prot.MpcObj.GetNetworks().ResetNetworkLog()

	matrices := prot.DLProtocol()

	for i, _ := range matrices {
		log.LLvl1("Size of matrix", i, ":", len(matrices[i]))
	}
	prot.Prot.MpcObj.GetNetworks().PrintNetworkLog()
}
