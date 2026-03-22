// 188 of compressed LOC
// without minimum: 142
package king

import (
	"fmt"
	"github.com/BurntSushi/toml"
	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas-private/cryptobasics"
	"github.com/hhcho/sfgwas-private/gwas"
	"github.com/hhcho/sfgwas-private/libspindle"
	"go.dedis.ch/onet/v3/log"
	"gonum.org/v1/gonum/mat"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

// KingProtocolInfo contains protocol specific infos and link to generic protocol structure
type KingProtocolInfo struct {
	Prot          *ProtocolInfo
	Data          [][]float64 // hashing tables - samples - snps
	ComparisonMap map[string][]int
	Minimum       bool
	Sqrt          bool
}

// ConfigKingProtocol parameters to configure the protocol
type ConfigKingProtocol struct {
	DataPath        string           `toml:"simple_data_path"`
	Separator       string           `toml:"separator"`
	NumberOfRows    int              `toml:"number_of_rows"`
	NumberOfColumns int              `toml:"number_of_columns"`
	OutDir          string           `toml:"output_dir"`
	CacheDir        string           `toml:"cache_dir"`
	StartingIndex   int              `toml:"starting_index"`
	ComparisonMap   map[string][]int `toml:"comparison_map"`
	Minimum         bool             `toml:"minimum"`
	Sqrt            bool             `toml:"sqrt"`
}

// InitializeKingProtocol initialized the protocol parameters with global and local config files, and reads the input data
func InitializeKingProtocol(pid int, configFolder string) (relativeProt *KingProtocolInfo) {

	ConfigKingGlob := new(ConfigKingProtocol)
	if _, err := toml.DecodeFile(filepath.Join(configFolder, "configGlobal.toml"), ConfigKingGlob); err != nil {
		fmt.Println(err)
		return nil
	}

	ConfigKing := new(ConfigKingProtocol)
	if _, err := toml.DecodeFile(filepath.Join(configFolder, fmt.Sprintf("configLocal.Party%d.toml", pid)), ConfigKing); err != nil {
		fmt.Println(err)
		return nil
	}

	// Create cache/output directories
	if err := os.MkdirAll(ConfigKing.CacheDir, 0755); err != nil {
		panic(err)
	}
	if err := os.MkdirAll(ConfigKing.OutDir, 0755); err != nil {
		panic(err)
	}

	data := make([][]float64, ConfigKing.NumberOfRows)
	if pid > 0 {
		readData, err := LoadDataset(ConfigKing.DataPath, []rune(ConfigKing.Separator)[0], false)
		if err != nil {
			log.Fatal(err)
		}
		// for now we use one example dataset, parties can read a subset of it
		data = readData[ConfigKing.StartingIndex : ConfigKing.StartingIndex+ConfigKing.NumberOfRows]
		log.LLvl1(len(data), len(data[0]))
		for j := 0; j < ConfigKing.NumberOfRows; j++ {
			data[j] = data[j][:ConfigKing.NumberOfColumns]
		}
		log.LLvl1(pid, " has data with dims:", len(data), len(data[0]))
	}

	prot := InitializeProtocol(pid, configFolder)

	return &KingProtocolInfo{
		Data:          data,
		Prot:          prot,
		ComparisonMap: ConfigKingGlob.ComparisonMap,
		Minimum:       ConfigKingGlob.Minimum,
		Sqrt:          ConfigKingGlob.Sqrt,
	}
}

// KingProtocol computes kinship coefficients among all paiwise values between matrices held by different parties
func (pi *KingProtocolInfo) KingProtocol() map[int][][]float64 {
	log.LLvl1(time.Now(), "Finished Setup")

	pid := pi.Prot.MpcObj[0].GetPid()
	cps := pi.Prot.Cps
	//var mutex sync.Mutex
	var wg sync.WaitGroup

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

	listPartiesSent := make([]int, 0)
	// send encrypted data to requested pids, which node performs which computation is defined by a map [node_doing_computation]: [list of nodes sending data to this node]
	log.LLvl1(pid, "prepares data to send to ", pi.ComparisonMap)
	for i, v := range pi.ComparisonMap {
		wg.Add(1)
		go func(i string, v []int) {
			defer wg.Done()
			if pid > 0 {
				otherPid, _ := strconv.Atoi(i)
				if otherPid != pid {
					for j := 0; j < len(v); j++ {
						if v[j] == pid {
							log.LLvl1(pid, " sends its data to ", otherPid)
							listPartiesSent = append(listPartiesSent, otherPid)

							// prepares local data to send
							// Y
							rowsLocal := len(pi.Data)
							cols := len(pi.Data[0])
							rowsOther := allPidsNbrRows[otherPid]
							Y := mat.NewDense(rowsLocal, cols, Flatten(pi.Data))
							//if pi.Sqrt {
							//	Y.Scale(0.1, Y)
							//}
							// Compute 1 \times Y^T \cdot Y^T
							// prepare matrix of ones // TODO function
							ones := make([]float64, rowsOther*cols)
							for o := 0; o < rowsOther*cols; o++ {
								ones[o] = 1
							}
							One := mat.NewDense(rowsOther, cols, ones)
							OneYTYT := mat.NewDense(rowsOther, rowsLocal, nil)
							YTYT := mat.DenseCopyOf(Y.T())
							// 1 \times Y^T \cdot Y^T
							YTYT.MulElem(Y.T(), Y.T())
							OneYTYT.Mul(One, YTYT)

							// compute matrix of number of ones per row (i.e., nbr of heterozygous), for denominator of king kinship coeff.
							hetY := mat.NewDense(rowsOther, rowsLocal, nil)
							for c := 0; c < rowsLocal; c++ {
								hetRow := 0.0
								for r := 0; r < cols; r++ {
									if Y.At(c, r) == 1 {
										hetRow = hetRow + 1
									}
								}
								if hetRow == 0 { // avoid division by zero
									hetRow = 1
								}
								for r := 0; r < rowsOther; r++ {
									hetY.Set(r, c, -1.0/hetRow)
								}
							}

							// TODO REMOVE, JUST FOR L2-HEFACTORY COMMUNICATION
							//pi.Prot.MpcObj.GetNetworks().ResetNetworkLog()
							//timeStart := time.Now()
							//dummy := mat.NewDense(8192, 32, nil)
							//dummyEncrypted := cryptobasics.EncryptDense(cps, mat.DenseCopyOf(dummy))
							//pi.Prot.MpcObj.GetNetworks()[otherPid].SendInt(len(dummyEncrypted), otherPid)
							//pi.Prot.MpcObj.GetNetworks()[otherPid].SendInt(len(dummyEncrypted[0]), otherPid)
							//pi.Prot.MpcObj.GetNetworks()[otherPid].SendCipherMatrix(dummyEncrypted, otherPid)
							//log.LLvl1("Time to send all matrix: ", time.Since(timeStart))
							//pi.Prot.MpcObj.GetNetworks().PrintNetworkLog()
							//log.Fatal()

							yrows, ycols := Y.Dims()
							OneYTYTrows, OneYTYTcols := OneYTYT.Dims()
							hetYrows, hetYcols := hetY.T().Dims()
							log.LLvl1(pid, " dimension check: ", yrows, ycols, " and ", OneYTYTrows, OneYTYTcols, " and ", hetYrows, hetYcols)
							// TODO send matrix function
							// send Y
							YEncrypted := cryptobasics.EncryptDense(cps, mat.DenseCopyOf(Y.T()))
							pi.Prot.MpcObj.GetNetworks()[otherPid].SendInt(len(YEncrypted), otherPid)
							pi.Prot.MpcObj.GetNetworks()[otherPid].SendInt(len(YEncrypted[0]), otherPid)
							pi.Prot.MpcObj.GetNetworks()[otherPid].SendCipherMatrix(YEncrypted, otherPid)

							// send OneYTYT
							OneYTYTEncrypted := cryptobasics.EncryptDense(cps, mat.DenseCopyOf(OneYTYT))
							pi.Prot.MpcObj.GetNetworks()[otherPid].SendInt(len(OneYTYTEncrypted), otherPid)
							pi.Prot.MpcObj.GetNetworks()[otherPid].SendInt(len(OneYTYTEncrypted[0]), otherPid)
							pi.Prot.MpcObj.GetNetworks()[otherPid].SendCipherMatrix(OneYTYTEncrypted, otherPid)
							// send hetY
							var hetYEncrypted libspindle.CipherMatrix
							if pi.Minimum {
								hetYEncrypted = cryptobasics.EncryptDense(cps, hetY)
								pi.Prot.MpcObj.GetNetworks()[otherPid].SendInt(len(hetYEncrypted), otherPid)
								pi.Prot.MpcObj.GetNetworks()[otherPid].SendInt(len(hetYEncrypted[0]), otherPid)
								pi.Prot.MpcObj.GetNetworks()[otherPid].SendCipherMatrix(hetYEncrypted, otherPid)
							}

							// after sending the data, this node is ready to help in collective operations (// TODO for more than 2 parties, we will need go routines to not block)
							log.LLvl1(pid, " sent everything, ready to decrypt for debugging")
							_, _ = pi.Prot.MpcObj.GetNetworks()[otherPid].CollectiveDecryptMat(cps, nil, otherPid)

							if pi.Minimum || pi.Sqrt {
								log.LLvl1(pid, " help for sign test", len(OneYTYTEncrypted[0]), len(OneYTYTEncrypted)) // TODO function
								distance := cryptobasics.CZeroMat(cps, len(OneYTYTEncrypted[0]), len(OneYTYTEncrypted))
								log.LLvl1(pid, ": start minimum computation", len(distance), len(distance[0]))
								//concatCvec := mpc_core.RVec{}
								//log.LLvl1(" [DEBUG] start CMatToSS ", pi.Prot.MpcObj[otherPid].GetRType(),
								//	distance, otherPid, len(distance), len(distance[0]), allPidsNbrRows[otherPid])
								localThresSS := pi.Prot.MpcObj[otherPid].CMatToSS(cps, pi.Prot.MpcObj[otherPid].GetRType(),
									distance, otherPid, len(distance), len(distance[0]), allPidsNbrRows[otherPid])

								// TODO: there is probably a better way of doing this scaling
								prec := pi.Prot.MpcObj[0].GetDataBits()
								testScale := pi.Prot.MpcObj[0].GetRType().Zero().FromFloat64(1.0, prec)
								var backToRmat mpc_core.RMat
								if pi.Minimum {
									backToRmat = pi.Prot.MpcObj[otherPid].IsPositiveMat(localThresSS)
									backToRmat.MulScalar(testScale)
									backToRmat = pi.Prot.MpcObj[0].TruncMat(backToRmat, pi.Prot.MpcObj[0].GetDataBits(), pi.Prot.MpcObj[0].GetFracBits())
								} else {
									backToRmat, _ = pi.Prot.MpcObj[otherPid].SqrtAndSqrtInverseMat(localThresSS, true)
								}

								_ = pi.Prot.MpcObj[otherPid].SSToCMat(cps, backToRmat)

								log.LLvl1(pid, " finished helping for sign test, ready to decrypt")
								_, _ = pi.Prot.MpcObj.GetNetworks()[otherPid].CollectiveDecryptMat(cps, nil, otherPid)
							}
						}
					}
				}
			}
		}(i, v)
	}

	// auxiliary party helps for MPC computations
	if pid == 0 {
		for i, v := range pi.ComparisonMap {
			//wg.Add(1)
			//go func(i string, v []int) {
			//defer wg.Done()
			// TODO create and use function for sign test on Rmat
			comparingPid, _ := strconv.Atoi(i)
			distance := cryptobasics.CZeroMat(cps, int(math.Ceil(float64(allPidsNbrRows[comparingPid])/float64(cps.GetSlots()))), allPidsNbrRows[v[0]])
			if pi.Minimum || pi.Sqrt {
				log.LLvl1(pid, ": start minimum computation", len(distance), len(distance[0]))
				//log.LLvl1("[]start CMatToSS ", pi.Prot.MpcObj[comparingPid].GetRType(),
				//	distance, comparingPid, len(distance), len(distance[0]), allPidsNbrRows[comparingPid])
				localThresSS := pi.Prot.MpcObj[comparingPid].CMatToSS(cps, pi.Prot.MpcObj[comparingPid].GetRType(),
					distance, comparingPid, len(distance), len(distance[0]), allPidsNbrRows[comparingPid])

				// TODO: there is probably a better way of doing this scaling
				testScale := pi.Prot.MpcObj[0].GetRType().Zero().FromFloat64(1.0, pi.Prot.MpcObj[0].GetDataBits())
				var backToRmat mpc_core.RMat
				if pi.Minimum {
					backToRmat = pi.Prot.MpcObj[comparingPid].IsPositiveMat(localThresSS)
					backToRmat.MulScalar(testScale)
					backToRmat = pi.Prot.MpcObj[0].TruncMat(backToRmat, pi.Prot.MpcObj[0].GetDataBits(), pi.Prot.MpcObj[0].GetFracBits())
				} else {
					backToRmat, _ = pi.Prot.MpcObj[comparingPid].SqrtAndSqrtInverseMat(localThresSS, true)
				}

				_ = pi.Prot.MpcObj[comparingPid].SSToCMat(cps, backToRmat)
				log.LLvl1(pid, " finished helping, ready to decrypt")
				_, _ = pi.Prot.MpcObj.GetNetworks()[comparingPid].CollectiveDecryptMat(cps, nil, comparingPid)
			}

			//}(i, v)

		}
		//wg.Wait()
	}

	log.LLvl1("Ready to receive from ", pi.ComparisonMap[strconv.Itoa(pid)])

	comparisonResults := make(map[int][][]float64, 0)
	for i := 0; i < len(pi.ComparisonMap[strconv.Itoa(pid)]); i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()

			otherPid := pi.ComparisonMap[strconv.Itoa(pid)][i]
			log.LLvl1(pid, " ready to receive Y from ", otherPid)
			nrowsY := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveInt(otherPid)
			log.LLvl1("received nrowsY: ", nrowsY)
			ncolsCipherY := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveInt(otherPid)
			log.LLvl1("received ncolsCipherY: ", ncolsCipherY)
			Y := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveCipherMatrix(cps, nrowsY, ncolsCipherY, otherPid)
			log.LLvl1("Received Matrix Y (#rows=", len(Y), ")")

			log.LLvl1(pid, " ready to receive YSquare from ", otherPid)
			nrowsYSquare := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveInt(otherPid)
			log.LLvl1("received nrowsYSquare: ", nrowsYSquare)
			ncolsCipherYSquare := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveInt(otherPid)
			log.LLvl1("received ncolsCipherYSquare: ", ncolsCipherYSquare)
			YSquare := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveCipherMatrix(cps, nrowsYSquare, ncolsCipherYSquare, otherPid)
			log.LLvl1("Received Matrix Y (#rows=", len(YSquare), ")")
			var Yhet libspindle.CipherMatrix
			if pi.Minimum {
				log.LLvl1(pid, " ready to receive Yhet from ", otherPid)
				nrowsYhet := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveInt(otherPid)
				log.LLvl1("received nrowsYhet: ", nrowsYhet)
				ncolsCipherYhet := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveInt(otherPid)
				log.LLvl1("received ncolsCipherYhet: ", ncolsCipherYhet)
				Yhet = pi.Prot.MpcObj.GetNetworks()[pid].ReceiveCipherMatrix(cps, nrowsYhet, ncolsCipherYhet, otherPid)
				log.LLvl1("Received Matrix Y (#Yhet=", len(Yhet), ")")
			}

			// prepare local data for comparison
			// X
			//rowsLocal := len(pi.Data) / 2 // TODO currently optimized for 2 parties
			//cols := len(pi.Data[0])
			//rowsOther := allPidsNbrRows[otherPid]
			//var X *mat.Dense
			//if pid == 1 { // TODO currently optimized for 2 parties
			//	X = mat.NewDense(rowsLocal, cols, Flatten(pi.Data[:rowsLocal]))
			//} else if pid == 2 {
			//	X = mat.NewDense(rowsLocal, cols, Flatten(pi.Data[rowsLocal:]))
			//}

			rowsLocal := len(pi.Data)
			cols := len(pi.Data[0])
			rowsOther := allPidsNbrRows[otherPid]
			var X *mat.Dense
			X = mat.NewDense(rowsLocal, cols, Flatten(pi.Data))

			// Compute 1 \times Y^T \cdot Y^T
			// prepare matrix of ones
			ones := make([]float64, cols*rowsOther)
			for o := 0; o < cols*rowsOther; o++ {
				ones[o] = 1
			}
			One := mat.NewDense(cols, rowsOther, ones)
			XXOne := mat.NewDense(rowsLocal, rowsOther, nil)
			XX := mat.DenseCopyOf(X)
			XX.MulElem(X, X)
			XXOne.Mul(XX, One)
			//log.LLvl1("XXOne ", mat.Formatted(XXOne))

			// compute matrix of number of ones per row (i.e., nbr of heterozygous)
			hetx := mat.NewDense(rowsLocal, rowsOther, nil)
			for r := 0; r < rowsLocal; r++ {
				hetRow := 0.0
				for c := 0; c < cols; c++ {
					if X.At(r, c) == 1 {
						hetRow = hetRow + 1
					}
				}
				if hetRow == 0 {
					hetRow = 1
				}
				for c := 0; c < rowsOther; c++ {
					hetx.Set(r, c, 1.0/hetRow)
				}
			}
			hetxEncoded := cryptobasics.EncodeDense(cps, hetx)

			Xrows, Xcols := X.Dims()
			XXOnerows, XXOnecols := XXOne.Dims()
			hetXrows, hetXcols := hetx.Dims()
			log.LLvl1(pid, " sends ", Xrows, Xcols, " and ", XXOnerows, XXOnecols, " and ", hetXrows, hetXcols)

			// compute kinship coefficients
			twoX := mat.DenseCopyOf(X.T())
			twoX.Scale(-2.0, X.T())
			twoXRow, twoXCol := twoX.Dims()
			log.LLvl1("MAT MULT ", len(Y), " x ", twoXRow, twoXCol)
			outLevel := 6
			if !pi.Minimum && !pi.Sqrt {
				outLevel = 5
			}
			twoXYT := gwas.CPMatMult4(cps, Y, twoX, outLevel)

			XSquareEncoded := cryptobasics.EncodeDense(cps, mat.DenseCopyOf(XXOne))
			distance := cryptobasics.AggregatePMat(cps, []libspindle.CipherMatrix{twoXYT, YSquare}, []libspindle.PlainMatrix{XSquareEncoded})

			distanceDecrypt, _ := pi.Prot.MpcObj[pid].Network.CollectiveDecryptMat(cps, distance, pid)
			decryptedMatrix := make([][]float64, len(distanceDecrypt))
			for d := 0; d < len(distanceDecrypt); d++ {
				decryptedMatrix[d] = libspindle.DecodeFloatVector(cps, distanceDecrypt[d])
				//log.LLvl1("localResultDecode : ", decryptedMatrix[d][:2*allPidsNbrRows[otherPid]])
			}
			comparisonResults[otherPid] = decryptedMatrix

			// compute minimum between hetX and hetY values for each sample comparison
			// with secret sharing (maximum of inverses) (here using MHE only might be more efficient)
			hetxMinushety := make(libspindle.CipherMatrix, len(Yhet))
			if pi.Minimum {
				log.LLvl1(pid, ": start minimum computation")

				for h := 0; h < len(Yhet); h++ {
					hetxMinushety[h] = cryptobasics.CPAdd(cps, Yhet[h], hetxEncoded[h])
				}
			} else if pi.Sqrt {
				hetxMinushety = distance
			}
			log.LLvl1(pid, " NETWORK BEFORE MINIMUM PART ", otherPid)
			pi.Prot.MpcObj.GetNetworks().PrintNetworkLog()
			if pi.Minimum || pi.Sqrt {
				log.LLvl1("start CMatToSS ", pi.Prot.MpcObj[pid].GetRType(),
					hetxMinushety, pid, len(hetxMinushety), len(hetxMinushety[0]), allPidsNbrRows[pid])
				hetxMinushetySS := pi.Prot.MpcObj[pid].CMatToSS(cps, pi.Prot.MpcObj[pid].GetRType(),
					hetxMinushety, pid, len(hetxMinushety), len(hetxMinushety[0]), allPidsNbrRows[pid])

				// TODO: there is probably a better way of doing this
				prec := pi.Prot.MpcObj[0].GetDataBits()
				testScale := pi.Prot.MpcObj[0].GetRType().Zero().FromFloat64(1.0, prec)
				var backToRmat mpc_core.RMat
				if pi.Minimum {
					backToRmat = pi.Prot.MpcObj[pid].IsPositiveMat(hetxMinushetySS)
					backToRmat.MulScalar(testScale)
					backToRmat = pi.Prot.MpcObj[0].TruncMat(backToRmat, pi.Prot.MpcObj[0].GetDataBits(), pi.Prot.MpcObj[0].GetFracBits())
				} else {
					backToRmat, _ = pi.Prot.MpcObj[pid].SqrtAndSqrtInverseMat(hetxMinushetySS, true)
				}

				hetxMinushetySSsign := pi.Prot.MpcObj[pid].SSToCMat(cps, backToRmat)
				distance = hetxMinushetySSsign

				if pi.Minimum {

					// minimum between a and b, using sign test result r on (a-b), which return 0 for neg and 1 for pos
					// --> sign(a-b)*a + ((sign(a-b)-1)*(-b))
					onesForHet := make([]float64, rowsOther*rowsLocal)
					for o := 0; o < rowsOther*rowsLocal; o++ {
						onesForHet[o] = -1
					}
					onesHetDense := mat.NewDense(rowsLocal, rowsOther, onesForHet)
					onesHetEncoded := cryptobasics.EncodeDense(cps, onesHetDense)

					for s := 0; s < len(hetxMinushetySSsign); s++ {
						tmp := cryptobasics.CPAdd(cps, hetxMinushetySSsign[s], onesHetEncoded[s])
						tmp2 := cryptobasics.CMult(cps, tmp, Yhet[s])
						hetxMinushetySSsign[s] = cryptobasics.CPMult(cps, hetxMinushetySSsign[s], hetxEncoded[s])
						hetxMinushetySSsign[s] = cryptobasics.CAdd(cps, hetxMinushetySSsign[s], tmp2)
						distance[s] = cryptobasics.CMult(cps, hetxMinushetySSsign[s], distance[s])
					}
				}

				distanceDecrypt, _ = pi.Prot.MpcObj[pid].Network.CollectiveDecryptMat(cps, distance, pid)
				decryptedMatrix = make([][]float64, len(distanceDecrypt))
				for d := 0; d < len(distanceDecrypt); d++ {
					distanceDecode := libspindle.DecodeFloatVector(cps, distanceDecrypt[d])
					decryptedMatrix[d] = distanceDecode[:allPidsNbrRows[pid]]
					//log.LLvl1(decryptedMatrix[d])
				}
				comparisonResults[otherPid] = decryptedMatrix
			}
		}(i)
	}
	wg.Wait()

	return comparisonResults
}
