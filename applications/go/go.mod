module sequre-private

go 1.17

require (
	github.com/BurntSushi/toml v0.4.1
	github.com/hhcho/mpc-core v0.0.0-20220828210829-24cf7abd1073
	github.com/hhcho/sfgwas-private v0.0.0-00010101000000-000000000000
	github.com/ldsec/lattigo/v2 v2.4.0
	go.dedis.ch/onet/v3 v3.2.10
	gonum.org/v1/gonum v0.9.3
)

require (
	github.com/aead/chacha20 v0.0.0-20180709150244-8b13a72661da // indirect
	github.com/daviddengcn/go-colortext v1.0.0 // indirect
	github.com/fanliao/go-concurrentMap v0.0.0-20141114143905-7d2d7a5ea67b // indirect
	github.com/google/uuid v1.3.0 // indirect
	github.com/gorilla/websocket v1.4.2 // indirect
	github.com/hhcho/frand v1.3.1-0.20210217213629-f1c60c334950 // indirect
	github.com/ldsec/unlynx v1.4.3 // indirect
	github.com/montanaflynn/stats v0.6.6 // indirect
	go.dedis.ch/fixbuf v1.0.3 // indirect
	go.dedis.ch/kyber/v3 v3.0.13 // indirect
	go.dedis.ch/protobuf v1.0.11 // indirect
	go.etcd.io/bbolt v1.3.6 // indirect
	golang.org/x/crypto v0.0.0-20210921155107-089bfa567519 // indirect
	golang.org/x/sys v0.0.0-20211117180635-dee7805ff2e1 // indirect
	golang.org/x/xerrors v0.0.0-20200804184101-5ec99f83aff1 // indirect
	rsc.io/goversion v1.2.0 // indirect
)

replace github.com/ldsec/lattigo/v2 => github.com/hcholab/lattigo/v2 v2.1.2-0.20230123224332-e8d68c24b94a

replace github.com/hhcho/sfgwas-private => ../../../../libs/go/src/hhcho/sfgwas-private
