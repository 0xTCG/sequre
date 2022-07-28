#include <seqan3/alphabet/nucleotide/dna5.hpp>
#include <seqan3/core/debug_stream.hpp>
#include <seqan3/search/views/kmer_hash.hpp>
 
using namespace seqan3::literals;
 
int main()
{
    std::vector<seqan3::dna5> text{"ACGNTAGCACGNTAGCACGNTAGC"_dna5};
 
    seqan3::debug_stream << (text | seqan3::views::kmer_hash(seqan3::ungapped{19})) << '\n';
 
    auto sigma_dna5 = seqan3::alphabet_size<seqan3::dna5>;
    std::cout << static_cast<uint16_t>(sigma_dna5) << '\n'; // 5
}
