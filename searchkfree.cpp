#include <iostream>  // for std::cout
#include <vector>    // for constructing vectors
#include <sstream>   // for reading comma-separated variables
#include <string>    // ""
#include <chrono>    // for timekeeping
#include <algorithm> // for std::remove_if
#include <cctype>    // for std::isspace
#include <cmath>     // for exponentiation
#include <array>     // for vectors of fixed (compile-time) length

using namespace std;

/// I/O FUNCTIONS //////////////////////////////////////////////////////////////////

// Remove all whitespace from a string
string remove_whitespace(string str)
{
    str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());
    return str;
}

// Convert an input string of the form [a,b,c,...] to a vector, eliminating spaces.
// Accepts various data types (e.g. int and double).
template <typename T>
vector<T> string_to_vector(string str)
{
    vector<T> out_vec;

    str = remove_whitespace(str);
    if (str.size() > 2 && str.front() == '[' && str.back() == ']')
    {
        // Strip the braces:
        string braceless_str = str.substr(1, str.size() - 2);

        stringstream str_stream(braceless_str);
        string item;

        // Comma-split and convert types:
        while (getline(str_stream, item, ','))
        {
            stringstream str_to_expr(item);
            T x;
            if (str_to_expr >> x)
            {
                out_vec.push_back(x);
            }
        }
    }
    return out_vec;
}

// Print a vector {v1,v2,v3,..} as [v1, v2, v3,...]
template <typename T>
void print(const vector<T> &vec)
{
    cout << "[";
    auto actual_delim = ", ";
    auto delim = "";

    for (const auto &elem : vec)
    {
        cout << delim << elem;
        delim = actual_delim;
    }
    cout << "]";
}

/// HELPER FUNCTIONS ///////////////////////////////////////////////////////////////

// Determine if the set S is len-free mod p.
// By "induction", suppose that S is len-free mod p if last element is removed.
// To avoid binary search, access S directly and through a characteristic function.
bool modpAPfreeQ(const vector<int> &S, array<bool, 400> &char_S, int p, int len, int t)
{
    // For the sake of membership testing, consider t as an element of S:
    char_S[t] = true;
    // Look for progressions in S containing t and with difference delta:
    // Loop through s in S, setting delta = t - s.
    for (int i = S.size() - 1; i >= 0; i--)
    {
        int s = S[i];
        int delta = t - s; // Note that delta in [1, p-2].

        // Try to extend the progression {s, t} past s; next term is s_ext:
        int found_len = 2;
        int s_ext;
        if (s < delta)
        {
            s_ext = s - delta + p;
        }
        else
        {
            s_ext = s - delta;
        }
        while (found_len < len)
        {
            if (!char_S[s_ext])
            {
                break;
            }
            // Update s_ext -= delta % p (using cases to avoid %):
            if (s_ext < delta)
            {
                s_ext += p - delta;
            }
            else
            {
                s_ext -= delta;
            }
            found_len++;
        }
        // We could check here if found_len >= len and return false, but it's faster on average to assume that found_len < len and prepare the next loop.

        // Try to extend the progression {s, last} past last; next term is s_ext:
        if (t + delta < p)
        {
            s_ext = t + delta;
        }
        else
        {
            s_ext = t + delta - p;
        }
        while (found_len < len)
        {
            if (!char_S[s_ext])
            {
                break;
            }
            // Update s_ext += delta % p (using cases to avoid %)
            if (s_ext + delta < p)
            {
                s_ext += delta;
            }
            else
            {
                s_ext -= p - delta;
            }
            found_len++;
        }
        // Exit early if an arithmetic progression of length len is found:
        if (found_len >= len)
        {
            char_S[t] = false;
            return false;
        }
    }
    char_S[t] = false;
    return true;
}

// Explicit instructions for computing x^0, x^1, x^2, x^3, x^4, x^5, and x^6.
double pow_small(double x, int y)
{
    switch (y)
    {
    case 0:
        return 1.0;
    case 1:
        return x;
    case 2:
        return x * x;
    case 3:
        return x * x * x;
    case 4:
    {
        double x2 = x * x;
        return x2 * x2;
    }
    case 5:
    {
        double x2 = x * x;
        return x2 * x2 * x;
    }
    case 6:
    {
        double x2 = x * x;
        return x2 * x2 * x2;
    }
    default:
        return pow(x, y);
    }
}

/// 2ND ORDER HARMONIC SUMS ////////////////////////////////////////////////////////

// Compute a (shifted) harmonic sum of 2-digit elements with base-p digits in S.
// Complexity is O(#S^2) divisions. Currently called only when printing to display.
double harmonic2(const vector<int> &S, int p)
{
    double hsum = 0.0;
    for (const auto &s : S)
    {
        double sp_plus = s * p + 1.0;
        for (const auto &t : S)
        {
            hsum += 1.0 / (sp_plus + t);
        }
    }
    return hsum;
}

// Compute a (shifted) harmonic sum of 2-digit elements with digits in S union T.
// We access S and T separately to avoid forming the concatenation S cat T.
// Currently called only to handle edge cases when #S + #T < 10.
double harmonic2pair(const vector<int> &S, const vector<int> &T, int p)
{
    double hsum = 0.0;
    for (const auto &s1 : S)
    {
        double s1p_plus = s1 * p + 1.0;
        for (const auto &s0 : S)
        {
            hsum += 1.0 / (s1p_plus + s0);
        }
        for (const auto &t0 : T)
        {
            hsum += 1.0 / (s1p_plus + t0);
        }
    }
    for (const auto &t1 : T)
    {
        double t1p_plus = t1 * p + 1.0;
        for (const auto &s0 : S)
        {
            hsum += 1.0 / (t1p_plus + s0);
        }
        for (const auto &t0 : T)
        {
            hsum += 1.0 / (t1p_plus + t0);
        }
    }
    return hsum;
}

// Same as before, but truncate S union T at a maximal size.
// Assumes full use of S, i.e. that S.size() <= maxsize.
// Currently called only to handle edge cases when #S + #T < 10 or maxsize < 10.
double harmonic2pairmaxsize(const vector<int> &S, const vector<int> &T, int maxsize, int p)
{
    size_t num_T_used = min(T.size(), maxsize - S.size());

    double hsum = 0.0;
    for (const auto &s1 : S)
    {
        double s1p_plus = s1 * p + 1.0;
        for (const auto &s0 : S)
        {
            hsum += 1.0 / (s1p_plus + s0);
        }
        for (size_t j = 0; j < num_T_used; ++j)
        {
            hsum += 1.0 / (s1p_plus + T[j]);
        }
    }
    for (size_t i = 0; i < num_T_used; ++i)
    {
        double t1p_plus = T[i] * p + 1.0;
        for (const auto &s0 : S)
        {
            hsum += 1.0 / (t1p_plus + s0);
        }
        for (size_t j = 0; j < num_T_used; ++j)
        {
            hsum += 1.0 / (t1p_plus + T[j]);
        }
    }
    return hsum;
}

/// 2ND ORDER HARMONIC SUM APPROXIMATION (VIA BAILLIE-SCHMELZER) ///////////////////

// Create tables caching [1/p, -1/p^2, 1/p^3...] and [0, 1, 2^6, 3^6, ..., p^6]
// This reduces mulitplications in bs2pair, etc.
array<double, 6> inv_p_table; // headsize = 6
vector<double> pow_6_table;   // Resized and populated in start_dfs.

// Estimate the B-S series for the 2-digit sum with digits in S union T.
// We access S and T separately to avoid forming the concatenation S cat T.
// For efficiency, uses precomputed power sums carried around in the state vector.
double bs2headpair(const array<double, 6> &S_minus_sums, const array<double, 6> &S_plus_sums, const vector<int> &T)
{
    double hsum = 0.0;
    // Copy the S_minus and S_plus power sum vectors:
    array<double, 6> ST_minus_sums = S_minus_sums;
    array<double, 6> ST_plus_sums = S_plus_sums;

    // Update these power sums based on the contribution of T:
    ST_plus_sums[0] += T.size();
    for (auto &t : T)
    {
        double tinv = 1 / double(t);
        double tinv_power = tinv * tinv;
        double tplus = t + 1.0;
        double tplus_power = tplus;

        ST_minus_sums[0] += tinv;
        for (size_t i = 1; i < 5; ++i) // headsize - 1 = 6 - 1 = 5
        {
            ST_minus_sums[i] += tinv_power;
            ST_plus_sums[i] += tplus_power;
            tinv_power *= tinv;
            tplus_power *= tplus;
        }
        ST_minus_sums[5] += tinv_power;
        ST_plus_sums[5] += tplus_power;
    }
    // Approximate the Baillie-Schmelzer series for the two-digit sum:
    for (int i = 0; i < 6; ++i)
    {
        hsum += ST_minus_sums[i] * ST_plus_sums[i] * inv_p_table[i];
    }
    return hsum;
}

// Sum the head of the Baillie-Schmelzer series for harmonic2pair(S, T, p).
// We access S and T separately to avoid forming the concatenation S cat T.
// For efficiency, use precomputed power sums related to S held in the state.
// Uses only num_T_used elements of T.
double bs2headpairmaxsize(const array<double, 6> &S_minus_sums, const array<double, 6> &S_plus_sums, const vector<int> &T, size_t num_T_used)
{
    double hsum = 0.0;
    // Copy the S_minus and S_plus power sum vectors:
    array<double, 6> ST_minus_sums = S_minus_sums;
    array<double, 6> ST_plus_sums = S_plus_sums;

    // Update these power sums based on the contribution of T:
    ST_plus_sums[0] += num_T_used;
    for (size_t j = 0; j < num_T_used; ++j)
    {
        double tinv = 1 / double(T[j]);
        double tinv_power = tinv * tinv;
        double tplus = T[j] + 1.0;
        double tplus_power = tplus;

        ST_minus_sums[0] += tinv;
        for (size_t i = 1; i < 5; ++i) // headsize - 1 = 6 - 1 = 5
        {
            ST_minus_sums[i] += tinv_power;
            ST_plus_sums[i] += tplus_power;
            tinv_power *= tinv;
            tplus_power *= tplus;
        }
        ST_minus_sums[5] += tinv_power;
        ST_plus_sums[5] += tplus_power;
    }
    // Approximate the Baillie-Schmelzer series for the two-digit sum:
    for (int i = 0; i < 6; ++i)
    {
        hsum += ST_minus_sums[i] * ST_plus_sums[i] * inv_p_table[i];
    }
    return hsum;
}

// Estimate the harmonic sum of 2-digit elements with base-p digits in S union T.
// If #S + #T < 10, do so exactly. Else, use an approximation to Baillie-Schmelzer.
double bs2pair(const vector<int> &S, const array<double, 6> &S_minus_sums, const array<double, 6> &S_plus_sums, double S_harmonic, const vector<int> &T, int p)
{
    // For small S or S union T, compute naively:
    if (S.size() + T.size() < 10 or S.size() < 2)
    {
        return harmonic2pair(S, T, p);
    }
    else
    {
        // Contribution of the single-digit sum:
        double hsum = S_harmonic;
        for (const auto &t : T)
        {
            hsum += 1.0 / (t + 1.0);
        }
        // Contribution of the series head:
        hsum += bs2headpair(S_minus_sums, S_plus_sums, T);
        // Contribution of the series tail:
        // This is hard-coded for headsize = 6. In general, replace 6 -> headsize and multiply tail_common_factor by (-1)^headsize.
        double denom_const = p * S[1];
        double tail_common_factor = pow_small(1 / denom_const, 6);
        int T_tail_index = T.size() - 1;
        while (T_tail_index >= 0 && 2 * T[T_tail_index] > p)
        {
            int t = T[T_tail_index];
            hsum += tail_common_factor * pow_6_table[1 + t] / (1.0 + denom_const + t);
            --T_tail_index;
        }
        // If T didn't fill out the tail, consider S:
        // TODO: Cache this contribution in the state vector?
        if (2 * T[0] > p)
        {
            int S_tail_index = S.size() - 1;
            while (2 * S[S_tail_index] > p)
            {
                int s = S[S_tail_index];
                hsum += tail_common_factor * pow_6_table[1 + s] / (1.0 + denom_const + s);
                --S_tail_index;
            }
        }
        return hsum;
    }
}

// Estimate the harmonic sum of 2-digit elements with base-p digits in S union T.
// Truncate T so that S union T has at most maxsize terms.
// If #S + #T < 10, do so exactly. Else, use an approximation to Baillie-Schmelzer.
// TODO: Profile cost of the introductory OR clause.
double bs2pairmaxsize(const vector<int> &S, const array<double, 6> &S_minus_sums, const array<double, 6> &S_plus_sums, double S_harmonic, const vector<int> &T, int p, int maxsize)
{
    // For small S or small maxsize, compute naively:
    if (S.size() + T.size() < 10 or maxsize < 10 or S.size() < 2)
    {
        return harmonic2pairmaxsize(S, T, maxsize, p);
    }
    else
    {
        // Contribution of the single-digit sum:
        double hsum = S_harmonic;
        size_t num_T_used = min(T.size(), maxsize - S.size());
        for (size_t j = 0; j < num_T_used; ++j)
        {
            hsum += 1.0 / (T[j] + 1.0);
        }
        // Contribution of the series head:
        hsum += bs2headpairmaxsize(S_minus_sums, S_plus_sums, T, num_T_used);
        // Contribution of the series tail:
        // This is hard-coded for headsize = 6. In general, replace 6 -> headsize and multiply tail_common_factor by (-1)^headsize.
        double denom_const = p * S[1];
        double tail_common_factor = pow_small(1 / denom_const, 6);

        int T_tail_index = num_T_used - 1; // TODO: Overwrite num_T_used to consolidate variables?
        while (T_tail_index >= 0 && 2 * T[T_tail_index] > p)
        {
            int t = T[T_tail_index];
            hsum += tail_common_factor * pow_6_table[1 + t] / (1.0 + denom_const + t);
            --T_tail_index;
        }
        // If T didn't fill out the tail, consider S:
        // TODO: Cache this contribution in the state vector?
        if (2 * T[0] > p)
        {
            int S_tail_index = S.size() - 1;
            while (2 * S[S_tail_index] > p)
            {
                int s = S[S_tail_index];
                hsum += tail_common_factor * pow_6_table[1 + s] / (1.0 + denom_const + s);
                --S_tail_index;
            }
        }
        return hsum;
    }
}

/// KEMPNER SET SUM ESTIMATION /////////////////////////////////////////////////////

// Create a global table to store cached values of (S_size / p)^beta.
// This vector will be resized to length p following declaration of p by user.
// This vector is populated during start_dfs().
vector<double> pow_beta_table;

// Estimate the harmonic sum of the (shifted) infinite set with base-p digits in S.
// Pass hsum2 from harmonic2, bs2pair, etc. for flexibility.
// Includes 3 best-fit parameters to improve linearity and monotonicity.
// Calls pow_beta_table to avoid repeated calls to pow().
double sumguess2(int S_size, int p, double hsum2, double alpha, double beta, double gamma)
{
    double sq_ratio = S_size * S_size / double(p * p);
    return ((1 + alpha * pow_beta_table[S_size]) * hsum2 - gamma * sq_ratio) / (1 - sq_ratio);
}

/// K-FREE SEARCH FUNCTIONS ////////////////////////////////////////////////////////

// Estimate the harmonic sum of a maximal k-free Kempner set.
// Print to terminal if fitness exceeds a given threshold.
// Uses harmonic2 for improved accuracy.
// TODO: Write a Baillie-Schmelzer variant to improve speed.
void evaluate(const vector<int> &S, int p, double thresh, double alpha, double beta, double gamma)
{
    double sg = sumguess2(S.size(), p, harmonic2(S, p), alpha, beta, gamma);
    if (sg >= thresh)
    {
        cout << "[" << p << ", " << sg << ", ";
        print(S);
        cout << "]\n";
    }
}

// As above, now tracking and outputting deviation data.
void evaluate_dev(const vector<int> &S, int p, size_t dev, double thresh, double alpha, double beta, double gamma)
{
    double sg = sumguess2(S.size(), p, harmonic2(S, p), alpha, beta, gamma);
    if (sg >= thresh)
    {
        cout << "[" << p << ", " << dev << ", " << sg << ", ";
        print(S);
        cout << "]\n";
    }
}

// Recursive function for depth-first search for len-free sets mod p. Parameters:
// S: current len-free set mod p;
// char_S: characteristic function of S, stored as a boolean array;
// T: set of possible t extending S;
// thresh: threshold for exploring/displaying with branch-and-bound;
// alpha, beta, gamma: 3 best-fit parameters for sumguess2().
// Caching S_minus_sums, S_plus_sums, and S_harmonic reduces scoring re-compute.
void dfs(vector<int> &S, array<bool, 400> &char_S, const vector<int> &T, int p, int len, array<double, 6> &S_minus_sums, array<double, 6> &S_plus_sums, double S_harmonic, double thresh, double alpha, double beta, double gamma)
{
    // If S is a maximal len-free set mod p, evaluate and backtrack:
    if (T.empty())
    {
        evaluate(S, p, thresh, alpha, beta, gamma);
        return;
    }

    // Backtrack if further progress is upper-bounded by some threshold:
    if (sumguess2(S.size() + T.size(), p, bs2pair(S, S_minus_sums, S_plus_sums, S_harmonic, T, p), alpha, beta, gamma) < thresh)
    {
        return;
    }

    // Otherwise, iterate over each possible extension to S:
    vector<int> T_prime; // Create a single T_prime per depth
    for (size_t i = 0; i < T.size(); ++i)
    {
        // Update S and char_S in place to avoid vector copying:
        int t = T[i];
        S.push_back(t);
        char_S[t] = true;

        // Adjust the S_minus_sums and S_plus_sums vectors:
        // Copying is faster than modifying/unmodifying in place.
        array<double, 6> S_minus_prime = S_minus_sums;
        array<double, 6> S_plus_prime = S_plus_sums;
        // TODO: Unroll this loop?
        double tinv = 1.0 / double(t);
        double tinv_power = tinv;
        double tplus = t + 1.0;
        double tplus_power = 1.0;
        for (size_t h = 0; h < 5; ++h) // headsize - 1 = 6 - 1 = 5
        {
            S_minus_prime[h] += tinv_power;
            S_plus_prime[h] += tplus_power;
            tinv_power *= tinv;
            tplus_power *= tplus;
        }
        S_minus_prime[5] += tinv_power;
        S_plus_prime[5] += tplus_power;

        // Clear T_prime from previous use in this loop:
        T_prime.clear();
        // Select from T those elements compatible with the new S:
        for (size_t j = i + 1; j < T.size(); ++j)
        {
            if (modpAPfreeQ(S, char_S, p, len, T[j]))
            {
                T_prime.push_back(T[j]);
            }
        }

        // Recur with the new state (S', T'):
        dfs(S, char_S, T_prime, p, len, S_minus_prime, S_plus_prime, S_harmonic + 1 / (1.0 + t), thresh, alpha, beta, gamma);

        // Undo changes to S and char_S:
        S.pop_back();
        char_S[t] = false;
    }
}

// Recursive call for depth-first search for len-free sets mod p. As before, with:
// maxsize: presumed upper bound for the size of a len-free subset mod p
// Use {p} to effectively ignore.
// The new parameter maxsize allows for much more efficient branch-and-bounding.
// Recommended setting is 1 larger than the largest-yet-discovered maxsize.
void dfs_maxsize(vector<int> &S, array<bool, 400> &char_S, const vector<int> &T, int p, int len, size_t maxsize, array<double, 6> &S_minus_sums, array<double, 6> &S_plus_sums, double S_harmonic, double thresh, double alpha, double beta, double gamma)
{
    // If S is a maximal len-free set mod p, evaluate and backtrack:
    if (T.empty())
    {
        evaluate(S, p, thresh, alpha, beta, gamma);
        return;
    }

    // Backtrack if further progress is upper-bounded by some threshold:
    if (sumguess2(min(S.size() + T.size(), maxsize), p, bs2pairmaxsize(S, S_minus_sums, S_plus_sums, S_harmonic, T, p, maxsize), alpha, beta, gamma) < thresh)
    {
        return;
    }

    // Otherwise, iterate over each possible extension to S:
    vector<int> T_prime; // Create a single T_prime per depth
    for (size_t i = 0; i < T.size(); ++i)
    {
        // Update S and char_S in place to avoid vector copying:
        int t = T[i];
        S.push_back(t);
        char_S[t] = true;

        // Adjust the S_minus_sums and S_plus_sums vectors:
        // Copying is faster than modifying/unmodifying in place.
        array<double, 6> S_minus_prime = S_minus_sums;
        array<double, 6> S_plus_prime = S_plus_sums;
        // TODO: Unroll this loop?
        double tinv = 1.0 / double(t);
        double tinv_power = tinv;
        double tplus = t + 1.0;
        double tplus_power = 1.0;
        for (size_t h = 0; h < 5; ++h) // headsize - 1 = 6 - 1 = 5
        {
            S_minus_prime[h] += tinv_power;
            S_plus_prime[h] += tplus_power;
            tinv_power *= tinv;
            tplus_power *= tplus;
        }
        S_minus_prime[5] += tinv_power;
        S_plus_prime[5] += tplus_power;

        // Clear T_prime from previous use in this loop:
        T_prime.clear();
        // Select from T those elements compatible with the new S:
        for (size_t j = i + 1; j < T.size(); ++j)
        {
            if (modpAPfreeQ(S, char_S, p, len, T[j]))
            {
                T_prime.push_back(T[j]);
            }
        }

        // Recur with the new state (S', T'):
        dfs_maxsize(S, char_S, T_prime, p, len, maxsize, S_minus_prime, S_plus_prime, S_harmonic + 1 / (1.0 + t), thresh, alpha, beta, gamma);

        // Undo changes to S and char_S:
        S.pop_back();
        char_S[t] = false;
    }
}

// Recursive call to depth-first search for len-free sets mod p. As before, with:
// devmax: maximal number of deviations from a greedy search
// Use {p} to effectively ignore.
// The total number of deviations (to that point) are included in the state vector.
void dfs_maxsize_dev(vector<int> &S, array<bool, 400> &char_S, const vector<int> &T, int p, int len, size_t maxsize, size_t dev, size_t devmax, array<double, 6> &S_minus_sums, array<double, 6> &S_plus_sums, double S_harmonic, double thresh, double alpha, double beta, double gamma)
{
    // If S is a maximal len-free set mod p, evaluate and backtrack:
    if (T.empty())
    {
        evaluate_dev(S, p, dev, thresh, alpha, beta, gamma);
        return;
    }

    // Backtrack if further progress is upper-bounded by some threshold:
    if (sumguess2(min(S.size() + T.size(), maxsize), p, bs2pairmaxsize(S, S_minus_sums, S_plus_sums, S_harmonic, T, p, maxsize), alpha, beta, gamma) < thresh)
    {
        return;
    }

    // Otherwise, iterate over each possible extension to S:
    vector<int> T_prime; // Create a single T_prime per depth
    // Limit the number of branches to devmax - dev.
    // This caps the total deviations at devmax - 1, not devmax.
    for (size_t i = 0; i < min(T.size(), devmax - dev); ++i)
    {
        // Update S and char_S in place to avoid vector copying:
        int t = T[i];
        S.push_back(t);
        char_S[t] = true;

        // Adjust the S_minus_sums and S_plus_sums vectors:
        // Copying is faster than modifying/unmodifying in place.
        array<double, 6> S_minus_prime = S_minus_sums;
        array<double, 6> S_plus_prime = S_plus_sums;
        // TODO: Unroll this loop?
        double tinv = 1.0 / double(t);
        double tinv_power = tinv;
        double tplus = t + 1.0;
        double tplus_power = 1.0;
        for (size_t h = 0; h < 5; ++h) // headsize - 1 = 6 - 1 = 5
        {
            S_minus_prime[h] += tinv_power;
            S_plus_prime[h] += tplus_power;
            tinv_power *= tinv;
            tplus_power *= tplus;
        }
        S_minus_prime[5] += tinv_power;
        S_plus_prime[5] += tplus_power;

        // Clear T_prime from previous use in this loop:
        T_prime.clear();
        // Select from T those elements compatible with the new S:
        for (size_t j = i + 1; j < T.size(); ++j)
        {
            if (modpAPfreeQ(S, char_S, p, len, T[j]))
            {
                T_prime.push_back(T[j]);
            }
        }

        // Recur with the new state (S', T'):
        dfs_maxsize_dev(S, char_S, T_prime, p, len, maxsize, dev + i, devmax, S_minus_prime, S_plus_prime, S_harmonic + 1.0 / (t + 1.0), thresh, alpha, beta, gamma);

        // Undo changes to S and char_S:
        S.pop_back();
        char_S[t] = false;
    }
}

// Initialize and launch a minimal version of depth-first search.
void start_dfs(vector<int> root, int p, int len, int maxsize, int devmax, double thresh, double alpha, double beta, double gamma)
{
    // Prepare the characteristic set of S:
    array<bool, 400> char_root;
    char_root.fill(false); // Default initialization should set all bits false, but compiler optimization might make this behavior erratic.
    for (int s : root)
    {
        char_root[s] = true;
    }

    // Prepare the initial complement vector:
    vector<int> T;
    for (int t = root[root.size() - 1] + 1; t < p - 1; ++t)
    {
        if (modpAPfreeQ(root, char_root, p, len, t))
        {
            T.push_back(t);
        }
    }

    // Prepare arrays used to cache results related to scoring the (S,T) states.
    // Start with the contribution of 0:
    array<double, 6> root_minus_sums = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    array<double, 6> root_plus_sums = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    root_plus_sums[0] = root.size();
    // Update with contributions of non-zero elements of the root:
    for (int j = 1; j < root.size(); ++j)
    {
        double rinv = 1 / double(root[j]);
        double rinv_power = rinv;
        double rplus = root[j] + 1.0;
        double rplus_power = 1.0;

        for (size_t i = 0; i < 5; ++i) // headsize - 1 = 6 - 1 = 5
        {
            root_minus_sums[i] += rinv_power;
            root_plus_sums[i] += rplus_power;
            rinv_power *= rinv;
            rplus_power *= rplus;
        }
        root_minus_sums[5] += rinv_power;
        root_plus_sums[5] += rplus_power;
    }

    // Compute the (shifted) harmonic sum of the root:
    double root_harmonic = 0.0;
    for (int s : root)
    {
        root_harmonic += 1.0 / (s + 1.0);
    }

    // Cache signed powers of 1/p used in bs2headpair and bs2headpairmaxsize.
    inv_p_table[0] = 1.0 / double(p);
    for (int i = 1; i < 6; ++i)
    {
        inv_p_table[i] = -inv_p_table[i - 1] * inv_p_table[0];
    }

    // Cache calls to pow() used for sumguess2():
    pow_beta_table.resize(p);
    for (int i = 1; i < p; ++i)
    {
        pow_beta_table[i] = pow(i / double(p), beta);
    }

    // Cache 6th powers of integers up to p:
    pow_6_table.resize(p);
    for (int i = 1; i < p; ++i)
    {
        pow_6_table[i] = pow_small(i, 6);
    }

    // Launch a minimal version of depth-first-search:
    if (maxsize >= p - 1 && devmax >= p - 1)
    {
        dfs(root, char_root, T, p, len, root_minus_sums, root_plus_sums, root_harmonic, thresh, alpha, beta, gamma);
    }
    else if (devmax >= p - 1)
    {
        dfs_maxsize(root, char_root, T, p, len, maxsize, root_minus_sums, root_plus_sums, root_harmonic, thresh, alpha, beta, gamma);
    }
    else
    {
        dfs_maxsize_dev(root, char_root, T, p, len, maxsize, 0, devmax, root_minus_sums, root_plus_sums, root_harmonic, thresh, alpha, beta, gamma);
    }
}

/// DEFAULT TUNING PARAMETERS //////////////////////////////////////////////////////

vector<vector<double>> default_tuning_array = {
    {0.0, 1.0, 0.0},          // 0 (for padding - do not call base 0)
    {0.0, 1.0, 0.0},          // 1 (for padding - do not call base 1)
    {0.0, 1.0, 0.0},          // 2 (for padding - do not call base 2)
    {0.0, 1.0, 0.0},          // 3 (return hsum2, no elabroate curve fitting)
    {0.0, 1.0, 0.0},          // 4 (return hsum2, no elabroate curve fitting)
    {0.4494, 1.2830, 2.1537}, // 5
    {0.3706, 1.2328, 2.0375}, // 6
    {0.3220, 1.2071, 1.9544}, // 7
    {0.2885, 1.1923, 1.8919}, // 8
    {0.2634, 1.1836, 1.8405}, // 9
    {0.2440, 1.1786, 1.7989}, // 10
    {0.2284, 1.1762, 1.7635}, //
    {0.2157, 1.1755, 1.7335}, //
    {0.2051, 1.1760, 1.7078}, //
    {0.1961, 1.1774, 1.6855}, //
    {0.1883, 1.1795, 1.6654}, // 15
    {0.1810, 1.1815, 1.6453}, //
    {0.1746, 1.1797, 1.6294}, //
    {0.1686, 1.1772, 1.6133}, //
    {0.1632, 1.1747, 1.5994}, //
    {0.1594, 1.1704, 1.5932}, // 20
    {0.1548, 1.1666, 1.5803}, //
    {0.1509, 1.1634, 1.5708}, //
    {0.1469, 1.1598, 1.5583}, //
    {0.1438, 1.1563, 1.5514}, //
    {0.1413, 1.1524, 1.5470}, // 25
    {0.1387, 1.1490, 1.5407}, //
    {0.1361, 1.1465, 1.5337}, //
    {0.1336, 1.1443, 1.5264}, //
    {0.1319, 1.1409, 1.5243}, //
    {0.1297, 1.1382, 1.5180}, // 30
    {0.1275, 1.1369, 1.5110}, //
    {0.1259, 1.1348, 1.5080}, //
    {0.1242, 1.1328, 1.5035}, //
    {0.1226, 1.1308, 1.4990}, //
    {0.1211, 1.1301, 1.4955}, // 35
    {0.1196, 1.1280, 1.4913}, //
    {0.1182, 1.1275, 1.4870}, //
    {0.1164, 1.1256, 1.4798}, //
    {0.1150, 1.1249, 1.4752}, //
    {0.1139, 1.1241, 1.4724}, // 40
    {0.1132, 1.1206, 1.4728}, //
    {0.1117, 1.1209, 1.4663}, //
    {0.1110, 1.1195, 1.4660}, //
    {0.1103, 1.1180, 1.4658}, //
    {0.1094, 1.1161, 1.4639}, // 45
    {0.1087, 1.1141, 1.4635}, //
    {0.1080, 1.1138, 1.4626}, //
    {0.1069, 1.1123, 1.4579}, //
    {0.1066, 1.1111, 1.4597}, //
    {0.1063, 1.1104, 1.4623}, // 50
    {0.1038, 1.1117, 1.4456}, //
    {0.1038, 1.1085, 1.4493}, //
    {0.1028, 1.1086, 1.4450}, //
    {0.1019, 1.1073, 1.4415}, //
    {0.1007, 1.1080, 1.4356}, // 55
    {0.1010, 1.1069, 1.4417}, //
    {0.1002, 1.1060, 1.4384}, //
    {0.1007, 1.1022, 1.4462}, //
    {0.0999, 1.1022, 1.4433}, //
    {0.0983, 1.1038, 1.4321}, // 60
    {0.0989, 1.0995, 1.4414}, //
    {0.0983, 1.0981, 1.4390}, //
    {0.0982, 1.0963, 1.4418}, //
    {0.0963, 1.0998, 1.4278}, //
    {0.0956, 1.0984, 1.4250}, // 65
    {0.0951, 1.0996, 1.4229}, //
    {0.0938, 1.1015, 1.4141}, //
    {0.0943, 1.0984, 1.4224}, //
    {0.0941, 1.0950, 1.4233}, //
    {0.0935, 1.0971, 1.4205}, // 70
    {0.0928, 1.0953, 1.4172}, //
    {0.0929, 1.0938, 1.4201}, //
    {0.0922, 1.0944, 1.4167}, //
    {0.0918, 1.0924, 1.4158}, //
    {0.0929, 1.0895, 1.4286}, // 75
    {0.0910, 1.0916, 1.4130}, //
    {0.0914, 1.0897, 1.4196}, //
    {0.0901, 1.0918, 1.4099}, //
    {0.0909, 1.0874, 1.4202}, //
    {0.0895, 1.0913, 1.4091}, // 80
    {0.0905, 1.0874, 1.4209}, //
    {0.0883, 1.0911, 1.4019}, //
    {0.0913, 1.0814, 1.4329}, //
    {0.0890, 1.0875, 1.4133}, //
    {0.0885, 1.0882, 1.4102}, // 85
    {0.0905, 1.0788, 1.4319}, //
    {0.0863, 1.0900, 1.3939}, //
    {0.0877, 1.0841, 1.4098}, //
    {0.0883, 1.0824, 1.4168}, //
    {0.0858, 1.0859, 1.3952}, // 90
    {0.0846, 1.0882, 1.3858}, //
    {0.0877, 1.0787, 1.4176}, //
    {0.0869, 1.0822, 1.4119}, //
    {0.0865, 1.0799, 1.4100}, //
    {0.0849, 1.0829, 1.3963}, // 95
    {0.0857, 1.0803, 1.4053}, //
    {0.0860, 1.0784, 1.4106}, //
    {0.0857, 1.0779, 1.4096}, //
    {0.0843, 1.0809, 1.3970}, //
    {0.0852, 1.0767, 1.4085}, // 100
    {0.0852, 1.0771, 1.4094}, //
    {0.0842, 1.0779, 1.4013}, //
    {0.0834, 1.0797, 1.3955}, //
    {0.0842, 1.0774, 1.4050}, //
    {0.0846, 1.0744, 1.4108}, // 105
    {0.0858, 1.0702, 1.4246}, //
    {0.0855, 1.0695, 1.4227}, //
    {0.0842, 1.0732, 1.4117}, //
    {0.0834, 1.0745, 1.4052}, //
    {0.0839, 1.0712, 1.4119}, // 110
    {0.0830, 1.0743, 1.4045}, //
    {0.0822, 1.0747, 1.3973}, //
    {0.0837, 1.0709, 1.4141}, //
    {0.0838, 1.0689, 1.4164}, //
    {0.0834, 1.0689, 1.4139}, // 115
    {0.0797, 1.0806, 1.3783}, //
    {0.0816, 1.0729, 1.3985}, //
    {0.0795, 1.0766, 1.3787}, //
    {0.0788, 1.0799, 1.3725}, //
    {0.0808, 1.0736, 1.3948}, // 120
    {0.0812, 1.0710, 1.4001}, //
    {0.0806, 1.0708, 1.3953}, //
    {0.0779, 1.0785, 1.3690}, //
    {0.0787, 1.0776, 1.3785}, //
    {0.0811, 1.0688, 1.4048}, // 125
    {0.0805, 1.0694, 1.4001}, //
    {0.0776, 1.0779, 1.3707}, //
    {0.0806, 1.0679, 1.4036}, //
    {0.0799, 1.0669, 1.3977}, //
    {0.0806, 1.0656, 1.4057}, // 130
    {0.0788, 1.0686, 1.3890}, //
    {0.0783, 1.0723, 1.3849}, //
    {0.0801, 1.0646, 1.4042}, //
    {0.0807, 1.0626, 1.4121}, //
    {0.0791, 1.0655, 1.3964}, // 135
    {0.0804, 1.0621, 1.4115}, //
    {0.0780, 1.0703, 1.3875}, //
    {0.0804, 1.0611, 1.4142}, //
    {0.0776, 1.0685, 1.3853}, //
    {0.0792, 1.0648, 1.4032}, // 140
    {0.0754, 1.0759, 1.3644}, //
    {0.0786, 1.0617, 1.3994}, //
    {0.0768, 1.0690, 1.3814}, //
    {0.0781, 1.0610, 1.3967}, //
    {0.0765, 1.0686, 1.3805}, // 145
    {0.0799, 1.0574, 1.4182}, //
    {0.0735, 1.0784, 1.3505}, //
    {0.0769, 1.0629, 1.3886}, //
    {0.0721, 1.0801, 1.3382}, //
    {0.0757, 1.0687, 1.3771}, // 150
    {0.0783, 1.0586, 1.4066}, //
    {0.0784, 1.0579, 1.4085}, //
    {0.0794, 1.0554, 1.4201}, //
    {0.0754, 1.0645, 1.3787}, //
    {0.0751, 1.0671, 1.3762}, // 155
    {0.0766, 1.0606, 1.3927}, //
    {0.0772, 1.0561, 1.4010}, //
    {0.0741, 1.0680, 1.3681}, //
    {0.0770, 1.0571, 1.4009}, //
    {0.0771, 1.0555, 1.4023}, // 160
    {0.0761, 1.0628, 1.3928}, //
    {0.0754, 1.0601, 1.3866}, //
    {0.0729, 1.0723, 1.3595}, //
    {0.0761, 1.0598, 1.3953}, //
    {0.0753, 1.0584, 1.3878}, // 165
    {0.0787, 1.0472, 1.4259}, //
    {0.0755, 1.0569, 1.3914}, //
    {0.0747, 1.0609, 1.3842}, //
    {0.0730, 1.0654, 1.3661}, //
    {0.0737, 1.0636, 1.3752}, // 170
    {0.0723, 1.0664, 1.3602}, //
    {0.0721, 1.0654, 1.3593}, //
    {0.0726, 1.0628, 1.3653}, //
    {0.0744, 1.0576, 1.3864}, //
    {0.0724, 1.0656, 1.3653}, // 175
    {0.0734, 1.0612, 1.3767}, //
    {0.0730, 1.0614, 1.3728}, //
    {0.0745, 1.0591, 1.3906}, //
    {0.0733, 1.0613, 1.3786}, //
    {0.0717, 1.0652, 1.3611}, // 180
    {0.0731, 1.0586, 1.3781}, //
    {0.0730, 1.0603, 1.3770}, //
    {0.0730, 1.0583, 1.3780}, //
    {0.0738, 1.0575, 1.3875}, //
    {0.0712, 1.0630, 1.3602}, // 185
    {0.0717, 1.0643, 1.3660}, //
    {0.0722, 1.0609, 1.3722}, //
    {0.0703, 1.0643, 1.3519}, //
    {0.0724, 1.0566, 1.3766}, //
    {0.0705, 1.0626, 1.3559}, // 190
    {0.0703, 1.0632, 1.3541}, //
    {0.0729, 1.0527, 1.3846}, //
    {0.0690, 1.0692, 1.3415}, //
    {0.0737, 1.0493, 1.3949}, //
    {0.0700, 1.0635, 1.3547}, // 195
    {0.0712, 1.0568, 1.3687}, //
    {0.0710, 1.0613, 1.3668}, //
    {0.0704, 1.0625, 1.3609}, //
    {0.0774, 1.0348, 1.4407}, //
    {0.0722, 1.0569, 1.3820}, // 200
    {0.0760, 1.0390, 1.4257}, //
    {0.0699, 1.0629, 1.3582}, //
    {0.0726, 1.0496, 1.3896}, //
    {0.0700, 1.0623, 1.3601}, //
    {0.0717, 1.0530, 1.3803}, // 205
    {0.0730, 1.0495, 1.3962}, //
    {0.0720, 1.0522, 1.3847}, //
    {0.0676, 1.0661, 1.3365}, //
    {0.0687, 1.0650, 1.3493}, //
    {0.0716, 1.0532, 1.3826}, // 210
    {0.0700, 1.0588, 1.3651}, //
    {0.0686, 1.0603, 1.3506}, //
    {0.0721, 1.0523, 1.3903}, //
    {0.0716, 1.0509, 1.3858}, //
    {0.0673, 1.0668, 1.3377}, // 215
    {0.0695, 1.0568, 1.3633}, //
    {0.0710, 1.0527, 1.3809}, //
    {0.0722, 1.0423, 1.3947}, //
    {0.0715, 1.0466, 1.3881}, //
    {0.0691, 1.0575, 1.3616}, // 220
    {0.0700, 1.0538, 1.3715}, //
    {0.0691, 1.0568, 1.3622}, //
    {0.0689, 1.0548, 1.3601}, //
    {0.0714, 1.0509, 1.3895}, //
    {0.0698, 1.0532, 1.3722}, // 225
    {0.0709, 1.0493, 1.3849}, //
    {0.0697, 1.0527, 1.3719}, //
    {0.0693, 1.0534, 1.3688}, //
    {0.0681, 1.0580, 1.3554}, //
    {0.0718, 1.0438, 1.3983}, // 230
    {0.0715, 1.0458, 1.3952}, //
    {0.0703, 1.0478, 1.3826}, //
    {0.0690, 1.0530, 1.3684}, //
    {0.0709, 1.0432, 1.3907}, //
    {0.0685, 1.0548, 1.3633}, // 235
    {0.0692, 1.0510, 1.3725}, //
    {0.0672, 1.0593, 1.3489}, //
    {0.0697, 1.0455, 1.3789}, //
    {0.0698, 1.0457, 1.3804}, //
    {0.0665, 1.0587, 1.3428}, // 240
    {0.0672, 1.0572, 1.3514}, //
    {0.0673, 1.0577, 1.3530}, //
    {0.0669, 1.0569, 1.3492}, //
    {0.0668, 1.0553, 1.3492}, //
    {0.0671, 1.0548, 1.3525}, // 245
    {0.0682, 1.0503, 1.3664}, //
    {0.0713, 1.0404, 1.4027}, //
    {0.0688, 1.0481, 1.3742}, //
    {0.0716, 1.0368, 1.4077}, //
    {0.0632, 1.0681, 1.3101}, // 250
    {0.0750, 1.5160, 1.2166}, // Begin new parameter model, optimized for #S <= 100
    {0.0757, 1.5158, 1.2216}, //
    {0.0757, 1.5195, 1.2194}, //
    {0.0755, 1.5204, 1.2176}, //
    {0.0759, 1.5243, 1.2183}, // 255
    {0.0748, 1.5251, 1.2107}, //
    {0.0762, 1.5280, 1.2186}, //
    {0.0762, 1.5314, 1.2164}, //
    {0.0755, 1.5312, 1.2124}, //
    {0.0753, 1.5330, 1.2105}, // 260
    {0.0764, 1.5365, 1.2151}, //
    {0.0761, 1.5372, 1.2133}, //
    {0.0759, 1.5376, 1.2119}, //
    {0.0762, 1.5402, 1.2122}, //
    {0.0764, 1.5437, 1.2116}, // 265
    {0.0766, 1.5461, 1.2114}, //
    {0.0767, 1.5431, 1.2144}, //
    {0.0764, 1.5489, 1.2088}, //
    {0.0770, 1.5502, 1.2123}, //
    {0.0771, 1.5510, 1.2125}, // 270
    {0.0772, 1.5530, 1.2119}, //
    {0.0769, 1.5529, 1.2104}, //
    {0.0775, 1.5578, 1.2106}, //
    {0.0778, 1.5589, 1.2120}, //
    {0.0776, 1.5642, 1.2079}, // 275
    {0.0773, 1.5608, 1.2087}, //
    {0.0775, 1.5627, 1.2090}, //
    {0.0781, 1.5673, 1.2097}, //
    {0.0781, 1.5694, 1.2081}, //
    {0.0779, 1.5706, 1.2067}, // 280
    {0.0785, 1.5724, 1.2092}, //
    {0.0779, 1.5715, 1.2068}, //
    {0.0783, 1.5751, 1.2071}, //
    {0.0776, 1.5772, 1.2015}, //
    {0.0794, 1.5824, 1.2089}, // 285
    {0.0785, 1.5784, 1.2065}, //
    {0.0789, 1.5851, 1.2045}, //
    {0.0791, 1.5857, 1.2053}, //
    {0.0791, 1.5877, 1.2046}, //
    {0.0787, 1.5871, 1.2025}, // 290
    {0.0791, 1.5880, 1.2048}, //
    {0.0788, 1.5906, 1.2017}, //
    {0.0793, 1.5900, 1.2043}, //
    {0.0796, 1.5950, 1.2033}, //
    {0.0799, 1.5981, 1.2033}, // 295
    {0.0801, 1.5981, 1.2046}, //
    {0.0799, 1.5977, 1.2041}, //
    {0.0808, 1.6034, 1.2059}, //
    {0.0803, 1.5992, 1.2050}, //
    {0.0803, 1.6071, 1.2009}, // 300
    {0.0809, 1.6048, 1.2053}, //
    {0.0807, 1.6144, 1.1986}, //
    {0.0808, 1.6130, 1.1999}, //
    {0.0816, 1.6150, 1.2032}, //
    {0.0806, 1.6151, 1.1978}, // 305
    {0.0808, 1.6157, 1.1994}, //
    {0.0815, 1.6177, 1.2009}, //
    {0.0811, 1.6204, 1.1977}, //
    {0.0819, 1.6207, 1.2020}, //
    {0.0820, 1.6240, 1.2005}, // 310
    {0.0818, 1.6238, 1.1994}, //
    {0.0816, 1.6265, 1.1966}, //
    {0.0821, 1.6285, 1.1984}, //
    {0.0821, 1.6303, 1.1970}, //
    {0.0821, 1.6299, 1.1979}, // 315
    {0.0820, 1.6372, 1.1931}, //
    {0.0831, 1.6390, 1.1977}, //
    {0.0825, 1.6383, 1.1952}, //
    {0.0815, 1.6382, 1.1899}, //
    {0.0828, 1.6421, 1.1949}, // 320
    {0.0832, 1.6448, 1.1950}, //
    {0.0840, 1.6432, 1.2000}, //
    {0.0837, 1.6480, 1.1959}, //
    {0.0837, 1.6488, 1.1955}, //
    {0.0837, 1.6491, 1.1952}, // 325
    {0.0848, 1.6495, 1.2010}, //
    {0.0840, 1.6566, 1.1925}, //
    {0.0846, 1.6586, 1.1942}, //
    {0.0853, 1.6583, 1.1975}, //
    {0.0846, 1.6583, 1.1943}, // 330
    {0.0852, 1.6659, 1.1929}, //
    {0.0852, 1.6620, 1.1951}, //
    {0.0860, 1.6665, 1.1967}, //
    {0.0858, 1.6700, 1.1934}, //
    {0.0851, 1.6711, 1.1899}, // 335
    {0.0864, 1.6688, 1.1975}, //
    {0.0866, 1.6777, 1.1924}, //
    {0.0854, 1.6761, 1.1883}, //
    {0.0856, 1.6757, 1.1895}, //
    {0.0869, 1.6775, 1.1941}, // 340
    {0.0867, 1.6805, 1.1913}, //
    {0.0874, 1.6817, 1.1943}, //
    {0.0869, 1.6835, 1.1909}, //
    {0.0863, 1.6847, 1.1870}, //
    {0.0866, 1.6870, 1.1873}, // 345
    {0.0873, 1.6920, 1.1877}, //
    {0.0874, 1.6933, 1.1878}, //
    {0.0882, 1.6920, 1.1920}, //
    {0.0878, 1.6926, 1.1903}, //
    {0.0886, 1.6967, 1.1911}, // 350
    {0.0891, 1.6934, 1.1957}, //
    {0.0893, 1.7002, 1.1922}, //
    {0.0889, 1.6998, 1.1912}, //
    {0.0889, 1.7039, 1.1881}, //
    {0.0893, 1.7115, 1.1852}, // 355
    {0.0899, 1.7040, 1.1927}, //
    {0.0897, 1.7069, 1.1901}, //
    {0.0910, 1.7105, 1.1934}, //
    {0.0900, 1.7126, 1.1879}, //
    {0.0898, 1.7108, 1.1885}, // 360
    {0.0903, 1.7131, 1.1889}, //
    {0.0907, 1.7185, 1.1870}, //
    {0.0909, 1.7194, 1.1880}, //
    {0.0914, 1.7204, 1.1896}, //
    {0.0919, 1.7239, 1.1893}, // 365
    {0.0918, 1.7242, 1.1891}, //
    {0.0910, 1.7265, 1.1841}, //
    {0.0921, 1.7265, 1.1890}, //
    {0.0917, 1.7289, 1.1862}, //
    {0.0924, 1.7318, 1.1870}, // 370
    {0.0923, 1.7352, 1.1846}, //
    {0.0931, 1.7376, 1.1864}, //
    {0.0930, 1.7351, 1.1880}, //
    {0.0933, 1.7398, 1.1863}, //
    {0.0939, 1.7394, 1.1891}, // 375
    {0.0944, 1.7462, 1.1867}, //
    {0.0950, 1.7463, 1.1890}, //
    {0.0935, 1.7450, 1.1841}, //
    {0.0952, 1.7481, 1.1891}, //
    {0.0948, 1.7508, 1.1860}, // 380
    {0.0952, 1.7501, 1.1882}, //
    {0.0950, 1.7538, 1.1849}, //
    {0.0951, 1.7562, 1.1839}, //
    {0.0964, 1.7535, 1.1908}, //
    {0.0959, 1.7573, 1.1870}, // 385
    {0.0964, 1.7618, 1.1854}, //
    {0.0956, 1.7610, 1.1829}, //
    {0.0971, 1.7617, 1.1885}, //
    {0.0981, 1.7653, 1.1900}, //
    {0.0976, 1.7695, 1.1860}, // 390
    {0.0975, 1.7702, 1.1850}, //
    {0.0979, 1.7684, 1.1879}, //
    {0.0991, 1.7704, 1.1911}, //
    {0.0977, 1.7773, 1.1810}, //
    {0.0980, 1.7767, 1.1830}, // 395
    {0.0986, 1.7818, 1.1822}, //
    {0.0990, 1.7795, 1.1849}, //
    {0.0988, 1.7836, 1.1816}, //
    {0.0996, 1.7851, 1.1838}, //
    {0.1008, 1.7864, 1.1878}  // 400
};

/// MAIN ///////////////////////////////////////////////////////////////////////////

int main()
{
    cout << "Enter an integer k for which our Kempner sets are k-free: ";
    string k_string;
    getline(cin, k_string);
    int k;
    stringstream(k_string) >> k;

    cout << "Enter a positive integer modulus b: ";
    string b_string;
    getline(cin, b_string);
    int b;
    stringstream(b_string) >> b;

    string tuning_string;
    vector<double> tuning_vector;
    if (b <= default_tuning_array.size() - 1)
    {
        // For small bases, suggest tuning data
        vector<double> default_tuning_vector = default_tuning_array[b];

        cout << "Enter estimation parameters, or * for pre-computed parameters (";
        print(default_tuning_vector);
        cout << "): ";

        getline(cin, tuning_string);
        if (tuning_string == "*")
        {
            tuning_vector = default_tuning_vector;
        }
        else
        {
            tuning_vector = string_to_vector<double>(tuning_string);
        }
    }
    else if (b <= 400) // Current size of the char_S bool array; could be enlarged.
    {
        // If no suggestion is available, force user to specify
        cout << "Enter estimation parameters (e.g. [0.07, 1.0, 1.4]): ";
        getline(cin, tuning_string);
        tuning_vector = string_to_vector<double>(tuning_string);
    }
    else
    {
        cout << "Modulus b > 400 overflows hard-coded array<bool, 400> set at compile-time.";
        string quit;
        cin >> quit;
        cout << "quitting.";
        return 0;
    }
    double alpha = tuning_vector[0];
    double beta = tuning_vector[1];
    double gamma = tuning_vector[2];

    cout << "Enter a threshold (e.g. 4.2) to determine printed output: ";
    string thresh_string;
    getline(cin, thresh_string);
    double thresh;
    stringstream(thresh_string) >> thresh;

    cout << "Enter a root for the search space (e.g. [0, 1, 2]) [or * for the root [0]]: ";
    string root_string;
    vector<int> root_vector;
    getline(cin, root_string);
    if (root_string == "*")
    {
        root_vector = {0};
    }
    else
    {
        root_vector = string_to_vector<int>(root_string);
    }

    cout << "Enter an upper bound for the size of allowable digit sets [or * to ignore]: ";
    string maxsize_string;
    getline(cin, maxsize_string);

    int maxsize;
    if (maxsize_string == "*")
    {
        maxsize = b;
    }
    else
    {
        maxsize = stoi(maxsize_string);
    }

    cout << "Enter a maximal number of deviations from a greedy search [or * to ignore]: ";
    string dev_string;
    getline(cin, dev_string);

    int devmax;
    if (dev_string == "*")
    {
        devmax = b;
    }
    else
    {
        // Correcting the difference between < and <=.
        devmax = stoi(dev_string) + 1;
    }

    cout << "Searching for k-free sets mod b...\n\n";

    // Get starting time-point
    auto start = chrono::high_resolution_clock::now();

    // Perform the appropriate depth-first search:
    start_dfs(root_vector, b, k, maxsize, devmax, thresh, alpha, beta, gamma);

    // Get ending time-point
    auto stop = chrono::high_resolution_clock::now();
    // Get duration, using duration_cast method for formatting
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);

    int run_time = duration.count();
    int ms_time = (run_time % 60000);
    run_time = (run_time - ms_time) / 60000;
    double s_time = ms_time / 1000.0;
    int m_time = run_time % 60;
    run_time = (run_time - m_time) / 60;
    int h_time = run_time;

    cout << "\nSearch completed in " << h_time << " hours, " << m_time << " minutes, and " << s_time << " seconds.  Enter a string to quit.\n";

    string quit;
    cin >> quit;
    cout << "quitting.";

    return 0;
}
