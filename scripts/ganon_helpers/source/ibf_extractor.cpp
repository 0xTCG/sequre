#include <cereal/archives/binary.hpp>
#include <seqan3/core/debug_stream.hpp>
#include <seqan3/search/dream_index/interleaved_bloom_filter.hpp>
#include <seqan3/alphabet/nucleotide/dna5.hpp>
#include <seqan3/search/views/kmer_hash.hpp>

#include <fstream>
#include <iostream>

using namespace seqan3::literals;
 
int main()
{
    seqan3::interleaved_bloom_filter<> filter;
    std::ifstream              is( "../source/sample_bacteria.ibf", std::ios::binary );
    cereal::BinaryInputArchive archive( is );
    archive( filter );

    // std::ofstream wf("ibf.bin", std::ios::out | std::ios::binary);
    // if( !wf ) {
    //     std::cout << "Cannot open file!" << std::endl;
    //     return 0;
    // }

    // int counter = 0;
    // uint8_t value = 0, *buffer = &value;
    // for ( auto e : filter.raw_data() ) {
    //     ++counter;
    //     value += e;
    //     if (counter % 8 == 0) {
    //         wf.write((char *)buffer, 1);
    //         value = 0;
    //     }
    //     value <<= 1;
    // }

    // wf.close();
    // if( !wf.good() ) {
    //     std::cout << "Error occurred at writing time!" << std::endl;
    //     return 0;
    // }

    seqan3::debug_stream << filter.bin_size() << " " << filter.bin_count() << " " << filter.bit_size() << " " << filter.hash_function_count() << std::endl;
    auto agent = filter.membership_agent();
    std::vector<seqan3::dna5> text{"ACGNTAGCACGNTAGCACGNTAGC"_dna5};
    seqan3::debug_stream << agent.bulk_contains(1184622173257ULL) << '\n';
    seqan3::debug_stream << agent.bulk_contains(5923110866288ULL) << '\n';
    seqan3::debug_stream << agent.bulk_contains(10542068003319ULL) << '\n';
    seqan3::debug_stream << agent.bulk_contains(14563367360345ULL) << '\n';
    seqan3::debug_stream << agent.bulk_contains(15596377817352ULL) << '\n';
    seqan3::debug_stream << agent.bulk_contains(1687943774261ULL) << '\n';
        
    return 0;
}
