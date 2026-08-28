// Independent coverage check for the support-orbit enumeration used by
// six_voter_exact_finite_verifier.cpp.
//
// It directly enumerates every labeled antichain of 4, 5, or 6 nontrivial
// supports, filters the reduction conditions, and compares the labeled counts
// with the sum of orbit sizes of the canonical representatives.

#define main finite_verifier_main
#include "six_voter_exact_finite_verifier.cpp"
#undef main

long long direct_count_by_size[7] = {};
vector<int> support_universe;

void direct_antichain_search(int start, int target, vector<int>& chosen) {
    if ((int)chosen.size() == target) {
        int degree[6] = {};
        for (int column : chosen)
            for (int i = 0; i < 6; ++i)
                if ((column >> i) & 1) ++degree[i];
        for (int i = 0; i < 6; ++i)
            if (degree[i] < 2 || degree[i] > target - 1) return;
        if (!full_column_rank(chosen)) return;
        ++direct_count_by_size[target];
        return;
    }
    const int needed = target - (int)chosen.size();
    for (int index = start; index <= (int)support_universe.size() - needed; ++index) {
        const int candidate = support_universe[index];
        bool incomparable = true;
        for (int column : chosen) {
            if ((column & candidate) == column || (column & candidate) == candidate) {
                incomparable = false;
                break;
            }
        }
        if (!incomparable) continue;
        chosen.push_back(candidate);
        direct_antichain_search(index + 1, target, chosen);
        chosen.pop_back();
    }
}

int main(int argc, char** argv) {
    const string report_path = argc >= 2
        ? argv[1]
        : "six_voter_exact_coverage_report.txt";
    const auto started = chrono::steady_clock::now();

    for (int mask = 1; mask < 63; ++mask) {
        const int size = __builtin_popcount((unsigned)mask);
        if (2 <= size && size <= 5) support_universe.push_back(mask);
    }

    vector<int> chosen;
    for (int m = 4; m <= 6; ++m)
        direct_antichain_search(0, m, chosen);

    // Build the canonical representatives independently after resetting maps.
    permutation_maps.clear();
    const vector<vector<int>> representatives = reduced_support_representatives();
    long long orbit_sum[7] = {};
    int representative_count[7] = {};
    for (const auto& columns : representatives) {
        const int m = (int)columns.size();
        ++representative_count[m];
        int automorphisms = 0;
        vector<int> sorted_columns = columns;
        sort(sorted_columns.begin(), sorted_columns.end());
        for (const auto& permutation : permutation_maps) {
            vector<int> image;
            for (int column : columns) image.push_back(permutation[column]);
            sort(image.begin(), image.end());
            if (image == sorted_columns) ++automorphisms;
        }
        assert(automorphisms > 0 && 720 % automorphisms == 0);
        orbit_sum[m] += 720 / automorphisms;
    }

    for (int m = 4; m <= 6; ++m)
        assert(orbit_sum[m] == direct_count_by_size[m]);
    assert(direct_count_by_size[4] == 9745);
    assert(direct_count_by_size[5] == 129570);
    assert(direct_count_by_size[6] == 444716);

    ofstream report(report_path);
    report << "EXACT COVERAGE CHECK FOR THE SIX-VOTER SUPPORT ENUMERATION\n\n";
    report << "Arithmetic: integer only; direct labeled enumeration versus orbit-size sum.\n";
    for (int m = 4; m <= 6; ++m) {
        report << "m=" << m
               << ": labeled reduced antichains=" << direct_count_by_size[m]
               << ", canonical representatives=" << representative_count[m]
               << ", sum of orbit sizes=" << orbit_sum[m] << "\n";
    }
    report << "RESULT: PASS.  The canonical representative list covers every labeled reduced support system.\n";
    report.close();

    cerr << "PASS coverage. Runtime "
         << chrono::duration<double>(chrono::steady_clock::now() - started).count()
         << " seconds.\n";
    return 0;
}
