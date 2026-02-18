/*
 * Privacy Guardian
 */

/*
Infinite loop.
sends random values as private inputs (in batches)
Need the number of possible labels to add noise like in private
    -> probably the same if performed on the number of teacher
    -> thus we send batches of size nb_teachers instead of size_batchs
Need to get the number of teachers

1)
    PG -> CP [open socket ?]
// 2)
//     CP -> PG [number of teachers] X
3) (loop)
    PG -> CP [vector of noise of size batch_size perpetually]
*/

#include "Math/gfp.h"
#include "Math/gf2n.h"
#include "Networking/sockets.h"
#include "Networking/ssl_sockets.h"
#include "Tools/int.h"
#include "Math/Setup.h"
#include "Protocols/fake-stuff.h"

#include "Math/gfp.hpp"

#include <sodium.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <random>
#include <cmath>
#include <vector>

std::random_device rd;
std::mt19937 gen(rd());

template<class T>
void create_noise(std::vector<T> &noise, std::normal_distribution<double> distribution) {
    for (unsigned random_value = 0; random_value < noise.size(); ++random_value)
        noise[random_value] = T(int(distribution(gen)));
}


template<class T>
unsigned int get_number_teachers(vector<ssl_socket*>& sockets, int nparties) {
    vector<T> shares(3);
    octetStream os;
    for (int party = 0; party < nparties; party++)
    {
        os.reset_write_head();
        os.Receive(sockets[party]);
            for (unsigned int triple_element = 0; triple_element < 3; triple_element++) {
                T value;
                value.unpack(os);
                shares[triple_element] += value;
            }
    }
    if (T(shares[0] * shares[1]) != shares[2])
    {
        std::cerr << "Unable to authenticate the number of teachers as correct, aborting." << std::endl;
        throw mac_fail();
    }
    return shares[0];
}

// Send the private inputs masked with a random value.
// Receive shares of a preprocessed triple from each SPDZ engine, combine and check the triples are valid.
// Add the private input value to triple[0] and send to each spdz engine.
template<class T>
void send_private_noise(vector<ssl_socket*>& sockets, unsigned int nparties, unsigned int size_noise, std::normal_distribution<double> distribution)
{
    octetStream os;
    vector< vector<T> > triples(size_noise, vector<T>(3));
    vector<T> triple_shares(3);

    for (unsigned int party = 0; party < nparties; party++)
    {
        os.reset_write_head();
        os.Receive(sockets[party]);

        for (unsigned int i = 0; i < size_noise; i++)
        {
            for (int triple_element = 0; triple_element < 3; triple_element++)
            {
                triple_shares[triple_element].unpack(os);
                triples[i][triple_element] += triple_shares[triple_element];
            }
        }
    }

    // Check triple relations (is a party cheating?)
    for (unsigned int i = 0; i < size_noise; i++)
    {
        if (T(triples[i][0] * triples[i][1]) != triples[i][2])
        {
            cerr << triples[i][2] << " != " << triples[i][0] << " * " << triples[i][1] << endl;
            cerr << "Incorrect triple at " << i << ", aborting\n";
            throw mac_fail();
        }
    }

    std::vector<T> noise(size_noise);
    create_noise<T>(noise, distribution);

    // Send inputs + triple[0], so SPDZ can compute shares of each value
    os.reset_write_head();

    for (unsigned i = 0; i < size_noise; i++) {
        T y = noise[i] + triples[i][0];
        std::cout << "guardian : " << noise[i] << std::endl;
        y.pack(os);
    }
    for (unsigned int party = 0; party < nparties; party++)
        os.Send(sockets[party]);
}

template<class T>
void run(vector<ssl_socket*>& sockets, unsigned int nparties, unsigned int nteachers, unsigned int batch_size, normal_distribution<double> distribution)
{
    // unsigned int nteachers = get_number_teachers(sockets, nparties);

    for (unsigned int vote = 0; vote < batch_size; vote++)
        send_private_noise<T>(sockets, nparties, nteachers, distribution);
}


int main(int argc, char** argv)
{
    unsigned int batch_size, nparties, nteachers;
    int port_base = 14000;

    if (argc != 7) {
        cout << "Usage is " << argv[0] << " <number of spdz parties> <batch size>"
           << " <number of teachers> <hosts filename> <sigma> <nrounds>"<< endl;
        exit(0);
    }

    nparties = atoi(argv[1]);
    batch_size = atoi(argv[2]);
    nteachers = atoi(argv[3]);

    vector<string> aggregator_hosts;
    // Read the hosts file
    std::ifstream hosts_file(argv[4]);
    string host;
    if (hosts_file.is_open()) {
        for (unsigned int party = 0; party < nparties; party++) {
            hosts_file >> host;
            aggregator_hosts.push_back(host.substr(0, host.find(":"))); // removes the port (if specified)
        }
    } else {
        std::cout << "Issue: unable to open file of hosts /app_saferlearn/HOSTS"<< std::endl;
        return 1;
    }

    double sigma = strtod(argv[5], NULL);

    std::normal_distribution<double> distribution(0, sigma);

    int nrounds = atoi(argv[6]);
    int round = 0;

    std::cout << "parties " << nparties << " batch " << batch_size << " nteachers " << nteachers << " sigma " << sigma << " nrounds " << nrounds << std::endl;

    while (nrounds < 0 || round++ < nrounds){
        bigint::init_thread();

        // Setup connections from this client to each party socket
        vector<int> plain_sockets(nparties);
        vector<ssl_socket*> sockets(nparties);
        ssl_ctx ctx("C" + to_string(nteachers));
        ssl_service io_service;
        octetStream specification;
        for (unsigned int party = 0; party < nparties; party++)
        {
            set_up_client_socket(plain_sockets[party], aggregator_hosts[party].c_str(), port_base + party);

            send(plain_sockets[party], (octet*) &nteachers, sizeof(int));
            sockets[party] = new ssl_socket(io_service, ctx, plain_sockets[party],
                    "P" + to_string(party), "C" + to_string(nteachers), true);
            if (party == 0)
                specification.Receive(sockets[0]);
            octetStream os;
            os.store(2);
            os.Send(sockets[party]);
        }
        cout << "Finish setup socket connections of the Privacy Guardian to SPDZ engines." << endl;

        int type = specification.get<int>();
        switch (type)
        {
        case 'p':
        {
            gfp::init_field(specification.get<bigint>());
            cerr << "using prime " << gfp::pr() << endl;
            run<gfp>(sockets, nparties, nteachers, batch_size, distribution);
            break;
        }
        case 'R':
        {
            int R = specification.get<int>();
            switch (R)
            {
            case 64: {
                run<Z2<64>>(sockets, nparties, nteachers, batch_size, distribution);
                break;
            }
            case 104: {
                run<Z2<104>>(sockets, nparties, nteachers, batch_size, distribution);
                break;
            }
            case 128: {
                run<Z2<128>>(sockets, nparties, nteachers, batch_size, distribution);
                break;
            }
            default:
                cerr << R << "-bit ring not implemented";
                exit(1);
            }
            break;
        }
        default:
            cerr << "Type " << type << " not implemented";
            exit(1);
        }

        for (unsigned int i = 0; i < nparties; i++)
            delete sockets[i];
    }//end while

    return 0;
}
