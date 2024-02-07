package king

import (
	"fmt"
	"github.com/BurntSushi/toml"
	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas-private/libspindle"
	"github.com/hhcho/sfgwas-private/mpc"
	"github.com/ldsec/lattigo/v2/ckks"
	"go.dedis.ch/onet/v3/log"
	"path/filepath"
	"time"
)

type ProtocolConfig struct {
	NumMainParties int `toml:"num_main_parties"`
	HubPartyId     int `toml:"hub_party_id"`

	CkksParams string `toml:"ckks_params"`

	divSqrtMaxLen   int `toml:"div_sqrt_max_len"`
	Servers         map[string]mpc.Server
	MpcFieldSize    int    `toml:"mpc_field_size"`
	MpcDataBits     int    `toml:"mpc_data_bits"`
	MpcFracBits     int    `toml:"mpc_frac_bits"`
	MpcNumThreads   int    `toml:"mpc_num_threads"`
	LocalNumThreads int    `toml:"local_num_threads"`
	BindingIP       string `toml:"binding_ipaddr"`
}

type ProtocolInfo struct {
	MpcObj mpc.ParallelMPC
	Cps    *libspindle.CryptoParams
	Config *ProtocolConfig
}

func InitializeProtocol(pid int, configFolder string) (relativeProt *ProtocolInfo) {

	config := new(ProtocolConfig)
	// Import global parameters
	if _, err := toml.DecodeFile(filepath.Join(configFolder, "configGlobal.toml"), config); err != nil {
		fmt.Println(err)
		return
	}

	var chosen int
	switch config.CkksParams {
	case "PN12QP109":
		chosen = ckks.PN12QP109
	case "PN13QP218":
		chosen = ckks.PN13QP218
	case "PN14QP438":
		chosen = ckks.PN14QP438
	case "PN15QP880":
		chosen = ckks.PN15QP880
	case "PN16QP1761":
		chosen = ckks.PN16QP1761
	default:
		panic("Undefined value of CKKS params in Config")
	}

	params := ckks.DefaultParams[chosen]
	prec := uint(config.MpcFieldSize)
	networks := mpc.ParallelNetworks(mpc.InitCommunication(config.BindingIP, config.Servers, pid, config.NumMainParties+1, config.MpcNumThreads))
	for thread := range networks {
		networks[thread].SetMHEParams(params)
	}

	var rtype mpc_core.RElem
	switch config.MpcFieldSize {
	case 256:
		rtype = mpc_core.LElem256Zero
	case 128:
		rtype = mpc_core.LElem128Zero
	default:
		panic("Unsupported value of MPC field size")
	}

	log.LLvl1(fmt.Sprintf("MPC parameters: bit length %d, data bits %d, frac bits %d",
		config.MpcFieldSize, config.MpcDataBits, config.MpcFracBits))
	mpcEnv := mpc.InitParallelMPCEnv(networks, rtype, config.MpcDataBits, config.MpcFracBits)
	for thread := range mpcEnv {
		mpcEnv[thread].SetHubPid(config.HubPartyId)
	}

	//TODO
	log.LLvl1(time.Now(), "Debugging output: PRGs initial state")
	for i := 0; i <= config.NumMainParties; i++ {
		if i != pid {
			mpcEnv[0].Network.Rand.SwitchPRG(i)
		}
		r := mpcEnv[0].Network.Rand.RandElem(mpcEnv[0].GetRType())
		if i != pid {
			mpcEnv[0].Network.Rand.RestorePRG()
		}
		log.LLvl1(pid, i, ":", r)
	}

	cps := networks.CollectiveInit(params, prec)

	// TODO
	cv, _ := libspindle.EncryptFloatVector(cps, make([]float64, 1))
	d := cv[0].Value()[0].Coeffs
	log.LLvl1(time.Now(), "Enc check", d[0][0], d[1][1], d[2][2])

	return &ProtocolInfo{
		MpcObj: mpcEnv,
		Cps:    cps,
		Config: config,
	}
}
