// Exact finite verifier for the six-voter reduction.
//
// No floating-point arithmetic and no external optimizer are used.  The program
// enumerates all reduced approval-support systems up to voter permutations,
// all relevant integer utility targets, and exact cap-cover dual certificates.
// It also exactly enumerates the vertices of the open floor cells, obtaining the
// eight interior holes and checking their singleton punctures.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>
using namespace std;

long long determinant(vector<vector<long long>> a) {
    const int n = (int)a.size();
    if (n == 0) return 1;
    long long sign = 1, previous = 1;
    for (int k = 0; k < n - 1; ++k) {
        int pivot_row = k;
        while (pivot_row < n && a[pivot_row][k] == 0) ++pivot_row;
        if (pivot_row == n) return 0;
        if (pivot_row != k) {
            swap(a[pivot_row], a[k]);
            sign = -sign;
        }
        const long long pivot = a[k][k];
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                const long long numerator = a[i][j] * pivot - a[i][k] * a[k][j];
                assert(previous != 0 && numerator % previous == 0);
                a[i][j] = numerator / previous;
            }
            a[i][k] = 0;
        }
        previous = pivot;
    }
    return sign * a[n - 1][n - 1];
}

bool full_column_rank(const vector<int>& columns) {
    const int m = (int)columns.size();
    vector<int> selected_rows(6, 0);
    fill(selected_rows.begin(), selected_rows.begin() + m, 1);
    sort(selected_rows.begin(), selected_rows.end(), greater<int>());
    do {
        vector<int> rows;
        for (int i = 0; i < 6; ++i) if (selected_rows[i]) rows.push_back(i);
        vector<vector<long long>> matrix(m, vector<long long>(m));
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < m; ++c)
                matrix[r][c] = (columns[c] >> rows[r]) & 1;
        if (determinant(matrix) != 0) return true;
    } while (prev_permutation(selected_rows.begin(), selected_rows.end()));
    return false;
}

vector<array<unsigned char, 64>> permutation_maps;

uint64_t canonical_code(const vector<int>& columns) {
    const int m = (int)columns.size();
    uint64_t best = UINT64_MAX;
    vector<int> image(m);
    for (const auto& permutation : permutation_maps) {
        for (int j = 0; j < m; ++j) image[j] = permutation[columns[j]];
        sort(image.begin(), image.end());
        uint64_t code = 0;
        for (int j = 0; j < m; ++j) code = (code << 6) | image[j];
        best = min(best, code);
    }
    return best;
}

vector<int> decode_code(uint64_t code, int m) {
    vector<int> columns(m);
    for (int j = m - 1; j >= 0; --j) {
        columns[j] = (int)(code & 63ULL);
        code >>= 6;
    }
    return columns;
}

vector<vector<int>> reduced_support_representatives() {
    array<int, 6> permutation = {0, 1, 2, 3, 4, 5};
    do {
        array<unsigned char, 64> support_map{};
        for (int mask = 0; mask < 64; ++mask) {
            int image = 0;
            for (int i = 0; i < 6; ++i)
                if ((mask >> i) & 1) image |= 1 << permutation[i];
            support_map[mask] = (unsigned char)image;
        }
        permutation_maps.push_back(support_map);
    } while (next_permutation(permutation.begin(), permutation.end()));
    assert(permutation_maps.size() == 720);

    // Canonical augmentation.  Inductive completeness: deleting one column from
    // any (r+1)-antichain gives an r-antichain; after mapping that parent to its
    // canonical representative, the deleted column is one of the extensions below.
    vector<unordered_set<uint64_t>> level(7);
    for (int mask = 1; mask < 63; ++mask) {
        const int size = __builtin_popcount((unsigned)mask);
        if (2 <= size && size <= 5) level[1].insert(canonical_code({mask}));
    }
    for (int r = 1; r < 6; ++r) {
        for (uint64_t parent_code : level[r]) {
            const vector<int> parent = decode_code(parent_code, r);
            for (int mask = 1; mask < 63; ++mask) {
                const int size = __builtin_popcount((unsigned)mask);
                if (size < 2 || size > 5) continue;
                bool incomparable = true;
                for (int column : parent) {
                    if ((column & mask) == column || (column & mask) == mask) {
                        incomparable = false;
                        break;
                    }
                }
                if (!incomparable) continue;
                vector<int> child = parent;
                child.push_back(mask);
                level[r + 1].insert(canonical_code(child));
            }
        }
    }

    vector<vector<int>> representatives;
    for (int m = 4; m <= 6; ++m) {
        vector<uint64_t> codes(level[m].begin(), level[m].end());
        sort(codes.begin(), codes.end());
        for (uint64_t code : codes) {
            vector<int> columns = decode_code(code, m);
            int degree[6] = {};
            for (int column : columns)
                for (int i = 0; i < 6; ++i)
                    if ((column >> i) & 1) ++degree[i];
            bool nontrivial_rows = true;
            for (int i = 0; i < 6; ++i)
                if (degree[i] < 2 || degree[i] > m - 1) nontrivial_rows = false;
            if (nontrivial_rows && full_column_rank(columns))
                representatives.push_back(columns);
        }
    }
    return representatives;
}

struct DualVertex {
    // lambda_i = numerator_i / denominator.
    array<long long, 6> numerator{};
    long long denominator = 1;
    long long cap_penalty_numerator = 0;
};

vector<DualVertex> cap_cover_dual_vertices(const vector<int>& columns) {
    // Arrangement hyperplanes: lambda_i=0 and lambda(N(c))=1.
    vector<array<long long, 6>> normal;
    vector<long long> rhs;
    for (int i = 0; i < 6; ++i) {
        array<long long, 6> row{};
        row[i] = 1;
        normal.push_back(row);
        rhs.push_back(0);
    }
    for (int column : columns) {
        array<long long, 6> row{};
        for (int i = 0; i < 6; ++i) row[i] = (column >> i) & 1;
        normal.push_back(row);
        rhs.push_back(1);
    }

    const int hyperplanes = (int)normal.size();
    vector<int> selected(hyperplanes, 0);
    fill(selected.begin(), selected.begin() + 6, 1);
    sort(selected.begin(), selected.end(), greater<int>());

    set<tuple<array<long long, 6>, long long>> seen;
    vector<DualVertex> vertices;
    do {
        vector<int> active;
        for (int j = 0; j < hyperplanes; ++j) if (selected[j]) active.push_back(j);
        vector<vector<long long>> matrix(6, vector<long long>(6));
        vector<long long> right_hand_side(6);
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) matrix[r][c] = normal[active[r]][c];
            right_hand_side[r] = rhs[active[r]];
        }
        long long denominator = determinant(matrix);
        if (denominator == 0) continue;

        array<long long, 6> numerator{};
        for (int c = 0; c < 6; ++c) {
            auto replaced = matrix;
            for (int r = 0; r < 6; ++r) replaced[r][c] = right_hand_side[r];
            numerator[c] = determinant(replaced);
        }
        if (denominator < 0) {
            denominator = -denominator;
            for (auto& value : numerator) value = -value;
        }
        bool nonnegative = true;
        for (long long value : numerator) if (value < 0) nonnegative = false;
        if (!nonnegative) continue;

        long long divisor = denominator;
        for (long long value : numerator) divisor = gcd(divisor, llabs(value));
        denominator /= divisor;
        for (auto& value : numerator) value /= divisor;
        if (!seen.insert({numerator, denominator}).second) continue;

        DualVertex vertex;
        vertex.numerator = numerator;
        vertex.denominator = denominator;
        for (int column : columns) {
            long long load_numerator = 0;
            for (int i = 0; i < 6; ++i)
                if ((column >> i) & 1) load_numerator += numerator[i];
            vertex.cap_penalty_numerator += max(0LL, load_numerator - denominator);
        }
        vertices.push_back(vertex);
    } while (prev_permutation(selected.begin(), selected.end()));
    return vertices;
}

bool dominates(const array<int, 6>& utility, const array<int, 6>& target) {
    for (int i = 0; i < 6; ++i) if (utility[i] < target[i]) return false;
    return true;
}

struct MarginPoint {
    bool valid = false;
    long long denominator = 1;
    // Variables: f_0,...,f_{m-2}, epsilon.  f_{m-1} is eliminated.
    vector<long long> numerator;
};

MarginPoint exact_open_cell_maximum(
    const vector<int>& columns,
    int kappa,
    const array<int, 6>& floor_target
) {
    const int m = (int)columns.size();
    const int dimension = m;
    const int epsilon_index = dimension - 1;
    vector<vector<long long>> coefficient;
    vector<long long> rhs;
    auto add = [&](vector<long long> row, long long bound) {
        coefficient.push_back(row);
        rhs.push_back(bound);
    };

    { vector<long long> row(dimension); row[epsilon_index] = -1; add(row, 0); }
    for (int j = 0; j < m - 1; ++j) {
        vector<long long> row(dimension);
        row[j] = -1; row[epsilon_index] = 1; add(row, 0);       // f_j >= eps
        row.assign(dimension, 0);
        row[j] = 1; row[epsilon_index] = 1; add(row, 1);        // f_j <= 1-eps
    }
    {
        vector<long long> row(dimension);
        for (int j = 0; j < m - 1; ++j) row[j] = 1;
        row[epsilon_index] = 1;
        add(row, kappa);                                        // f_last >= eps
    }
    {
        vector<long long> row(dimension);
        for (int j = 0; j < m - 1; ++j) row[j] = -1;
        row[epsilon_index] = 1;
        add(row, 1 - kappa);                                    // f_last <= 1-eps
    }
    for (int i = 0; i < 6; ++i) {
        const int last_incidence = (columns[m - 1] >> i) & 1;
        vector<long long> row(dimension);
        for (int j = 0; j < m - 1; ++j)
            row[j] = -(((columns[j] >> i) & 1) - last_incidence);
        add(row, last_incidence * kappa - floor_target[i]);      // Af >= h

        row.assign(dimension, 0);
        for (int j = 0; j < m - 1; ++j)
            row[j] = ((columns[j] >> i) & 1) - last_incidence;
        row[epsilon_index] = 1;
        add(row, floor_target[i] + 1 - last_incidence * kappa);  // Af <= h+1-eps
    }

    const int constraints = (int)coefficient.size();
    vector<int> selected(constraints, 0);
    fill(selected.begin(), selected.begin() + dimension, 1);
    sort(selected.begin(), selected.end(), greater<int>());

    MarginPoint best;
    best.numerator.assign(dimension, 0);
    do {
        vector<int> active;
        for (int j = 0; j < constraints; ++j) if (selected[j]) active.push_back(j);
        vector<vector<long long>> matrix(dimension, vector<long long>(dimension));
        vector<long long> right_hand_side(dimension);
        for (int r = 0; r < dimension; ++r) {
            matrix[r] = coefficient[active[r]];
            right_hand_side[r] = rhs[active[r]];
        }
        long long denominator = determinant(matrix);
        if (denominator == 0) continue;
        vector<long long> numerator(dimension);
        for (int c = 0; c < dimension; ++c) {
            auto replaced = matrix;
            for (int r = 0; r < dimension; ++r)
                replaced[r][c] = right_hand_side[r];
            numerator[c] = determinant(replaced);
        }
        if (denominator < 0) {
            denominator = -denominator;
            for (auto& value : numerator) value = -value;
        }

        bool feasible = true;
        for (int r = 0; r < constraints; ++r) {
            __int128 left = 0;
            for (int c = 0; c < dimension; ++c)
                left += (__int128)coefficient[r][c] * numerator[c];
            if (left > (__int128)rhs[r] * denominator) {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;

        if (!best.valid ||
            (__int128)numerator[epsilon_index] * best.denominator >
            (__int128)best.numerator[epsilon_index] * denominator) {
            best.valid = true;
            best.denominator = denominator;
            best.numerator = numerator;
        }
    } while (prev_permutation(selected.begin(), selected.end()));
    return best;
}

string support_string(int mask) {
    string result;
    for (int i = 0; i < 6; ++i) if ((mask >> i) & 1) result += char('1' + i);
    return result;
}

int main(int argc, char** argv) {
    const string report_path = argc >= 2
        ? argv[1]
        : "six_voter_exact_finite_report.txt";
    const auto started = chrono::steady_clock::now();

    const vector<vector<int>> representatives = reduced_support_representatives();
    map<int, int> representatives_by_size;
    for (const auto& columns : representatives) ++representatives_by_size[(int)columns.size()];

    long long kernel_budget_cases = 0;
    long long integer_targets = 0;
    long long nonintegral_targets = 0;
    long long puncture_failure_targets = 0;
    long long puncture_failures_dually_excluded = 0;
    long long fractionally_feasible_holes = 0;
    long long interior_holes = 0;
    long long maximum_dual_vertices = 0;
    vector<string> interior_descriptions;

    for (const vector<int>& columns : representatives) {
        const int m = (int)columns.size();
        int degree[6] = {};
        for (int column : columns)
            for (int i = 0; i < 6; ++i)
                if ((column >> i) & 1) ++degree[i];
        const vector<DualVertex> dual_vertices = cap_cover_dual_vertices(columns);
        maximum_dual_vertices = max<long long>(maximum_dual_vertices, dual_vertices.size());

        for (int kappa = 2; kappa <= m - 2; ++kappa) {
            ++kernel_budget_cases;
            vector<array<int, 6>> committee_utility;
            vector<vector<int>> committee_columns;
            vector<int> selected(m, 0);
            fill(selected.begin(), selected.begin() + kappa, 1);
            sort(selected.begin(), selected.end(), greater<int>());
            do {
                array<int, 6> utility{};
                vector<int> chosen;
                for (int c = 0; c < m; ++c) if (selected[c]) {
                    chosen.push_back(c);
                    for (int i = 0; i < 6; ++i)
                        if ((columns[c] >> i) & 1) ++utility[i];
                }
                committee_utility.push_back(utility);
                committee_columns.push_back(chosen);
            } while (prev_permutation(selected.begin(), selected.end()));

            array<int, 6> target{};
            function<void(int)> enumerate_target = [&](int row) {
                if (row < 6) {
                    for (int value = 0; value < degree[row]; ++value) {
                        target[row] = value;
                        enumerate_target(row + 1);
                    }
                    return;
                }
                ++integer_targets;

                bool integrally_implementable = false;
                for (const auto& utility : committee_utility)
                    if (dominates(utility, target)) {
                        integrally_implementable = true;
                        break;
                    }
                if (integrally_implementable) return;
                ++nonintegral_targets;

                // Exact value of the capped covering LP by arrangement-vertex duality.
                long long best_numerator = 0;
                long long best_denominator = 1;
                bool first_vertex = true;
                for (const auto& vertex : dual_vertices) {
                    long long value_numerator = -vertex.cap_penalty_numerator;
                    for (int i = 0; i < 6; ++i)
                        value_numerator += vertex.numerator[i] * target[i];
                    if (first_vertex ||
                        (__int128)value_numerator * best_denominator >
                        (__int128)best_numerator * vertex.denominator) {
                        first_vertex = false;
                        best_numerator = value_numerator;
                        best_denominator = vertex.denominator;
                    }
                }
                assert(!first_vertex);
                const bool fractionally_feasible =
                    (__int128)best_numerator <= (__int128)kappa * best_denominator;

                int first_failed_puncture = -1;
                vector<vector<int>> puncture_witness(6);
                for (int voter = 0; voter < 6; ++voter) {
                    array<int, 6> punctured = target;
                    --punctured[voter];
                    bool found = false;
                    for (int q = 0; q < (int)committee_utility.size(); ++q) {
                        if (dominates(committee_utility[q], punctured)) {
                            found = true;
                            puncture_witness[voter] = committee_columns[q];
                            break;
                        }
                    }
                    if (!found && first_failed_puncture < 0) first_failed_puncture = voter;
                }

                if (first_failed_puncture >= 0) {
                    ++puncture_failure_targets;
                    // An exact dual value > kappa proves that this target cannot be
                    // witnessed by any fractional committee of budget kappa.
                    assert(!fractionally_feasible);
                    ++puncture_failures_dually_excluded;
                }
                if (!fractionally_feasible) return;
                ++fractionally_feasible_holes;

                const MarginPoint cell = exact_open_cell_maximum(columns, kappa, target);
                if (!cell.valid || cell.numerator[m - 1] <= 0) return;
                ++interior_holes;

                // Recover the complete fractional point, including the eliminated last coordinate.
                vector<long long> fractional_numerator(m);
                long long partial_sum = 0;
                for (int c = 0; c < m - 1; ++c) {
                    fractional_numerator[c] = cell.numerator[c];
                    partial_sum += fractional_numerator[c];
                }
                fractional_numerator[m - 1] = kappa * cell.denominator - partial_sum;

                // The exact enumeration should show that the interior witness has A f = h.
                for (int i = 0; i < 6; ++i) {
                    long long utility_numerator = 0;
                    for (int c = 0; c < m; ++c)
                        if ((columns[c] >> i) & 1)
                            utility_numerator += fractional_numerator[c];
                    assert(utility_numerator == target[i] * cell.denominator);
                }
                // Every singleton puncture must have an integral witness.
                assert(first_failed_puncture < 0);

                ostringstream description;
                description << "Hole " << interior_holes
                            << ": m=" << m << ", kappa=" << kappa << ", supports={";
                for (int c = 0; c < m; ++c) {
                    if (c) description << ',';
                    description << support_string(columns[c]);
                }
                description << "}, h=(";
                for (int i = 0; i < 6; ++i) {
                    if (i) description << ',';
                    description << target[i];
                }
                description << "), f=(";
                for (int c = 0; c < m; ++c) {
                    if (c) description << ',';
                    const long long g = gcd(llabs(fractional_numerator[c]), cell.denominator);
                    description << fractional_numerator[c] / g << '/' << cell.denominator / g;
                }
                description << ")\n  singleton-puncture witnesses:";
                for (int voter = 0; voter < 6; ++voter) {
                    description << " voter " << voter + 1 << " -> {";
                    for (int q = 0; q < (int)puncture_witness[voter].size(); ++q) {
                        if (q) description << ',';
                        description << support_string(columns[puncture_witness[voter][q]]);
                    }
                    description << '}';
                }
                interior_descriptions.push_back(description.str());
            };
            enumerate_target(0);
        }
    }

    assert(representatives_by_size[4] == 40);
    assert(representatives_by_size[5] == 305);
    assert(representatives_by_size[6] == 853);
    assert((int)representatives.size() == 1198);
    assert(kernel_budget_cases == 3209);
    assert(puncture_failure_targets == puncture_failures_dually_excluded);
    assert(fractionally_feasible_holes == 46);
    assert(interior_holes == 8);

    ofstream report(report_path);
    report << "EXACT FINITE VERIFICATION FOR THE SIX-VOTER REDUCTION\n\n";
    report << "Arithmetic: integer only; no floating point and no external optimizer.\n";
    report << "Reduced support orbits: m=4: " << representatives_by_size[4]
           << ", m=5: " << representatives_by_size[5]
           << ", m=6: " << representatives_by_size[6] << "\n";
    report << "Total reduced support orbits: " << representatives.size() << "\n";
    report << "Kernel-budget cases: " << kernel_budget_cases << "\n";
    report << "Integer targets checked: " << integer_targets << "\n";
    report << "Nonintegral targets: " << nonintegral_targets << "\n";
    report << "Targets with a failed singleton puncture: " << puncture_failure_targets << "\n";
    report << "All such targets exactly certified fractionally infeasible: "
           << puncture_failures_dually_excluded << "\n";
    report << "Fractionally feasible nonintegral targets: " << fractionally_feasible_holes << "\n";
    report << "Targets with a strictly interior floor-cell witness: " << interior_holes << "\n";
    report << "Maximum number of cap-cover dual arrangement vertices: "
           << maximum_dual_vertices << "\n\n";
    for (const string& description : interior_descriptions)
        report << description << "\n\n";
    report << "RESULT: PASS.  The exhaustive finite lemma used in the six-voter proof holds.\n";
    report.close();

    cerr << "PASS: " << representatives.size() << " support orbits, "
         << kernel_budget_cases << " kernel-budget cases, "
         << interior_holes << " interior holes. Runtime "
         << chrono::duration<double>(chrono::steady_clock::now() - started).count()
         << " seconds.\n";
    return 0;
}
