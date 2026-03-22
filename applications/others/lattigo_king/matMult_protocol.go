// 188 of compressed LOC
// without minimum: 142
package king

import (
        "fmt"
        "github.com/BurntSushi/toml"
        "github.com/hhcho/sfgwas-private/cryptobasics"
        "github.com/hhcho/sfgwas-private/gwas"
        "github.com/hhcho/sfgwas-private/libspindle"
        "go.dedis.ch/onet/v3/log"
        "gonum.org/v1/gonum/mat"
        "os"
        "path/filepath"
        "time"
)

// MatMultInfo contains protocol specific infos and link to generic protocol structure
type MatMultProtocolInfo struct {
        Prot *ProtocolInfo
        Data [][]float64
        ATA  bool
}

// ConfigKingProtocol parameters to configure the protocol
type ConfigMatMultProtocol struct {
        DataPath        string `toml:"simple_data_path"`
        Separator       string `toml:"separator"`
        NumberOfRows    int    `toml:"number_of_rows"`
        NumberOfColumns int    `toml:"number_of_columns"`
        OutDir          string `toml:"output_dir"`
        CacheDir        string `toml:"cache_dir"`
        ATA             bool   `toml:"ATA"`
}

// InitializeMatMultProtocol initialized the protocol parameters with global and local config files, and reads the input data
func InitializeMatMultProtocol(pid int, configFolder string) (matMultProt *MatMultProtocolInfo) {

        ConfigMatMultGlob := new(ConfigMatMultProtocol)
        if _, err := toml.DecodeFile(filepath.Join(configFolder, "configGlobal.toml"), ConfigMatMultGlob); err != nil {
                fmt.Println(err)
                return nil
        }

        ConfigMatMult := new(ConfigMatMultProtocol)
        if _, err := toml.DecodeFile(filepath.Join(configFolder, fmt.Sprintf("configLocal.Party%d.toml", pid)), ConfigMatMult); err != nil {
                fmt.Println(err)
                return nil
        }

        // Create cache/output directories
        if err := os.MkdirAll(ConfigMatMult.CacheDir, 0755); err != nil {
                panic(err)
        }
        if err := os.MkdirAll(ConfigMatMult.OutDir, 0755); err != nil {
                panic(err)
        }

        data := make([][]float64, ConfigMatMult.NumberOfRows)
        if pid > 0 {
                readData, err := LoadDataset(ConfigMatMult.DataPath, []rune(ConfigMatMult.Separator)[0], false)
                if err != nil {
                        log.Fatal(err)
                }
                // for now we use one example dataset, parties can read a subset of it
                data = readData[0:ConfigMatMult.NumberOfRows]
                for j := 0; j < ConfigMatMult.NumberOfRows; j++ {
                        data[j] = data[j][:ConfigMatMult.NumberOfColumns]
                }
                log.LLvl1(pid, " has data with dims:", len(data), len(data[0]))
        }

        prot := InitializeProtocol(pid, configFolder)

        return &MatMultProtocolInfo{
                Data: data,
                Prot: prot,
                ATA:  ConfigMatMultGlob.ATA,
        }
}

// KingProtocol computes kinship coefficients among all paiwise values between matrices held by different parties
func (pi *MatMultProtocolInfo) MatMultProtocol() [][]float64 {
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

        if pid > 0 {
                rowsLocal := len(pi.Data)
                cols := len(pi.Data[0])
                Y := mat.NewDense(rowsLocal, cols, Flatten(pi.Data))
                if pi.ATA {
                        YTY := mat.NewDense(cols, cols, nil)
                        YTY.Mul(Y.T(), Y)
                        log.LLvl1("Local Matrix computed")
                        YTYEnc := cryptobasics.EncryptDense(cps, mat.DenseCopyOf(YTY.T()))
                        log.LLvl1("Matrix encrypted")
                        result := pi.Prot.MpcObj[0].Network.AggregateCMat(cps, YTYEnc)
                        log.LLvl1("Aggregated")
                        resultDecrypt, _ := pi.Prot.MpcObj[1].Network.CollectiveDecryptMat(cps, result, -1)
                        decryptedMatrix := make([][]float64, len(resultDecrypt))
                        for d := 0; d < len(resultDecrypt); d++ {
                                decryptedMatrix[d] = libspindle.DecodeFloatVector(cps, resultDecrypt[d])
                        }
                        return decryptedMatrix
                } else {
                        if pid == 2 {

                                YTEnc := cryptobasics.EncryptDense(cps, mat.DenseCopyOf(Y.T()))

                                pi.Prot.MpcObj.GetNetworks()[1].SendInt(len(YTEnc), 1)
                                pi.Prot.MpcObj.GetNetworks()[1].SendInt(len(YTEnc[0]), 1)
                                pi.Prot.MpcObj.GetNetworks()[1].SendCipherMatrix(YTEnc, 1)
                                _, _ = pi.Prot.MpcObj[1].Network.CollectiveDecryptMat(cps, nil, 1)
                        } else {
                                nrowsY := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveInt(2)
                                ncolsCipherY := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveInt(2)
                                YT := pi.Prot.MpcObj.GetNetworks()[pid].ReceiveCipherMatrix(cps, nrowsY, ncolsCipherY, 2)

                                result := gwas.CPMatMult4(cps, YT, mat.DenseCopyOf(Y.T()), 6)

                                resultDecrypt, _ := pi.Prot.MpcObj[1].Network.CollectiveDecryptMat(cps, result, 1)
                                decryptedMatrix := make([][]float64, len(resultDecrypt))
                                for d := 0; d < len(resultDecrypt); d++ {
                                        decryptedMatrix[d] = libspindle.DecodeFloatVector(cps, resultDecrypt[d])
                                        log.LLvl1("localResultDecode : ", decryptedMatrix[d][:2*allPidsNbrRows[1]])
                                }
                                return decryptedMatrix

                        }
                }
        }
        return nil
}
