#include "Controllers.h"

using namespace hmma;

double ANNSetter::thisRandom(double average, double standardDeviation)
{
    std::normal_distribution<double> d{average, standardDeviation};
    return d(m_gen);
}
float random(int min, int max, int step)
{
    int range = (max - min)*step;
    float result = ((float)(rand()%range)/(float)step) + (float)min;
    return result;
}

std::vector<int> randomSequence(int min, int max)
{
    std::vector<int> result;
    std::vector<int> sequence(max - min);
    for(int i(0), Range(max - min); i < Range; i++)
        sequence[i] = i;
    for(int i(0), Range(max - min); i < Range; i++)
    {
        int x = random(min, max - i, 1);
        x = sequence[x];
        sequence[x] = sequence[Range - (i+1)];
        result.push_back(x);
    }
    return result;
}

hmma::DDMatrix propMat(hmma::DDMatrix& a, int size2)
{
    hmma::DDMatrix result(a.rows(), size2);
    for(int i(0); i < size2; i++)
    {
        result.set_column(a.get_column(0).begin(), i);
    }
    return result;
}

void project(hmma::DDMatrix& target, hmma::DDMatrix& original, brnm::mat::Range ri, brnm::mat::Range rj)
{
    target.resize(ri.Step(), rj.Step());
    for(int i(0), c(rj.Step()); i < c; i++)
    {
        VectorRange<double>::iterator it = original.get_column(rj.start + i).begin() + (int)ri.start;
        target.set_column(it, i);
    }
}

hmma::DDMatrix column(hmma::DDMatrix& original, std::size_t c)
{
    hmma::DDMatrix result(original.rows(), 1);
    result.set_column(original.get_column(c).begin(), 0);
    return result;
}
hmma::DDMatrix row(hmma::DDMatrix& original, std::size_t r)
{
    hmma::DDMatrix result(1, original.columns());
    result.set_row(original.get_row(r).begin(), 0);
    return result;
}

hmma::DDMatrix element_prod(hmma::DDMatrix& a, hmma::DDMatrix& b)
{
    hmma::DDMatrix result(a.rows(), a.columns());
    DDMatrix::col_iterator iterR = result.col_begin();
    DDMatrix::col_iterator iterB = b.col_begin();
    for(DDMatrix::col_iterator iterA = a.col_begin(); iterA != a.col_end(); iterA++, iterB++, iterR++)
        *iterR = (*iterA) * (*iterB);
    return result;
}

hmma::DDMatrix element_prod(hmma::DDMatrix& a, hmma::DDMatrix* b)
{
    hmma::DDMatrix result(a.rows(), a.columns());
    DDMatrix::col_iterator iterR = result.col_begin();
    DDMatrix::col_iterator iterB = b->col_begin();
    for(DDMatrix::col_iterator iterA = a.col_begin(); iterA != a.col_end(); iterA++, iterB++, iterR++)
        *iterR = (*iterA) * (*iterB);
    return result;
}

hmma::DDMatrix element_prod(hmma::DDMatrix* a, hmma::DDMatrix* b)
{
    hmma::DDMatrix result(a->rows(), a->columns());
    DDMatrix::col_iterator iterR = result.col_begin();
    DDMatrix::col_iterator iterB = b->col_begin();
    for(DDMatrix::col_iterator iterA = a->col_begin(); iterA != a->col_end(); iterA++, iterB++, iterR++)
        *iterR = (*iterA) * (*iterB);
    return result;
}

hmma::DDMatrix NRelement_prod(hmma::DDMatrix a, hmma::DDMatrix b)
{
    hmma::DDMatrix result(a.rows(), a.columns());
    DDMatrix::col_iterator iterR = result.col_begin();
    DDMatrix::col_iterator iterB = b.col_begin();
    for(DDMatrix::col_iterator iterA = a.col_begin(); iterA != a.col_end(); iterA++, iterB++, iterR++)
        *iterR = (*iterA) * (*iterB);
    return result;
}

hmma::DDMatrix NORelement_prod(hmma::DDMatrix a, hmma::DDMatrix &b)
{
    hmma::DDMatrix result(a.rows(), a.columns());
    DDMatrix::col_iterator iterR = result.col_begin();
    DDMatrix::col_iterator iterB = b.col_begin();
    for(DDMatrix::col_iterator iterA = a.col_begin(); iterA != a.col_end(); iterA++, iterB++, iterR++)
        *iterR = (*iterA) * (*iterB);
    return result;
}

hmma::DDMatrix NRPelement_prod(hmma::DDMatrix a, hmma::DDMatrix *b)
{
    hmma::DDMatrix result(a.rows(), a.columns());
    DDMatrix::col_iterator iterR = result.col_begin();
    DDMatrix::col_iterator iterB = b->col_begin();
    for(DDMatrix::col_iterator iterA = a.col_begin(); iterA != a.col_end(); iterA++, iterB++, iterR++)
        *iterR = (*iterA) * (*iterB);
    return result;
}
/*
****************************
ANN SETTER
****************************
*/

ANNSetter::ANNSetter(): m_gen(std::chrono::system_clock::now().time_since_epoch().count())
{

}

ANNSetter::~ANNSetter()
{

}

void ANNSetter::Init(ANN* ann, unsigned int flags)
{
    if((flags & INIT_ALL) != 0)
    {
        ann->m_vW.clear();
        if((flags & INIT_WEIGHT_LARGE) != 0)
        {
            for(int k(0); k<ann->m_nbrL-1; k++)
            {
                ann->m_vW.push_back(DDMatrix(ann->m_nbrN[k+1], ann->m_nbrN[k]));
                for(int i(0); i < ann->m_vW[k].rows(); i++)
                    for(int j(0); j < ann->m_vW[k].columns(); j++)
                        ann->m_vW[k](i, j) = thisRandom(0, 1*pow(10, PRECISION))/pow(10, PRECISION);
            }
        }
        else
        {
            for(int k(0); k<ann->m_nbrL-1; k++)
            {
                double NNumber(sqrt((double)ann->GetNeuronsNumber(k)));
                ann->m_vW.push_back(DDMatrix(ann->m_nbrN[k+1], ann->m_nbrN[k]));
                for(int i(0); i < ann->m_vW[k].rows(); i++)
                    for(int j(0); j < ann->m_vW[k].columns(); j++)
                        ann->m_vW[k](i, j) = thisRandom(0, 1*pow(10, PRECISION))/pow(10, PRECISION)/NNumber;
            }
        }

        for(int j(0); j< ann->m_nbrL-1; j++)
        {
            for(int i(0); i < ann->m_biais[j].rows(); i++)
                ann->m_biais[j](i, 0) = thisRandom(0, 1*pow(10, PRECISION))/pow(10, PRECISION);
        }
        return;
    }
    if((flags & INIT_WEIGHT_DEFAULT) != 0)
    {
        ann->m_vW.clear();
        for(int k(0); k<ann->m_nbrL-1; k++)
        {
            double NNumber(sqrt((double)ann->GetNeuronsNumber(k)));
            ann->m_vW.push_back(DDMatrix(ann->m_nbrN[k+1], ann->m_nbrN[k]));
            for(int i(0); i < ann->m_vW[k].rows(); i++)
                for(int j(0); j < ann->m_vW[k].columns(); j++)
                    ann->m_vW[k](i, j) = thisRandom(0, 1*pow(10, PRECISION))/pow(10, PRECISION)/NNumber;
        }
    }
    if((flags & INIT_WEIGHT_LARGE) != 0)
    {
        ann->m_vW.clear();
        for(int k(0); k<ann->m_nbrL-1; k++)
        {
            ann->m_vW.push_back(DDMatrix(ann->m_nbrN[k+1], ann->m_nbrN[k]));
            for(int i(0); i < ann->m_vW[k].rows(); i++)
                for(int j(0); j < ann->m_vW[k].columns(); j++)
                    ann->m_vW[k](i, j) = thisRandom(0, 1*pow(10, PRECISION))/pow(10, PRECISION);
        }
}
    if((flags & INIT_BIAIS) != 0)
    {
        for(int j(0); j< ann->m_nbrL-1; j++)
        {
            for(int i(0); i < ann->m_biais[j].rows(); i++)
                ann->m_biais[j](i, 0) = thisRandom(0, 1*pow(10, PRECISION))/pow(10, PRECISION);
        }
    }
    if((flags & USE_PERSONAL_FUNCTION_FOR_BIAIS) != 0)
    {
        m_initBiais(ann->m_biais);
    }
    if((flags & USE_PERSONAL_FUNCTION_FOR_WEIGHT) != 0)
    {
        m_initWeight(ann->m_vW);
    }
    if((flags & USE_PERSONAL_FUNCTION_FOR_ALL) != 0)
    {
        m_initAll(ann->m_vW, ann->m_biais);
    }
}


// setters
void ANNSetter::SetInitAllFonction(int (*func)(std::vector<DDMatrix>& , std::vector<DDMatrix>& ))
{
    m_initAll = func;
}
void ANNSetter::SetInitBiaisFonction(int (*func)(std::vector<DDMatrix>& ))
{
    m_initBiais = func;
}
void ANNSetter::SetInitWeightFonction(int (*func)(std::vector<DDMatrix>& ))
{
    m_initWeight = func;
}

/*
****************************
ANN TRAINER
****************************
*/

// Constructor
ANNTrainer::ANNTrainer(): m_memorySize(10)
{
    m_arules = NORM;
    m_rtype = NONE;
    TOT_D_biais.clear();
    TOT_D_valueW.clear();

    m_valueW.clear();
    m_biais.clear();

    m_perf.clear();
    m_costMemory.clear();
}

// Destructor
ANNTrainer::~ANNTrainer()
{
    TOT_D_biais.clear();
    TOT_D_valueW.clear();

    m_valueW.clear();
    m_biais.clear();

    m_perf.clear();
    m_costMemory.clear();
}

void ANNTrainer::train(ANN* ann, double eta, unsigned int epoche, double lambda, double minAccuracy, int NonProgressionStopStep, int flags, double mu, PRINTTYPE printtype, unsigned int printStep)
{
    void (*funct)(double const&, double const&, double const&, double const&, DDMatrix&, DDMatrix&);
    void (*functS)(double const&, double const&, double const& , double const&, double const&, DDMatrix&, DDMatrix&, DDMatrix&);
    if((ENABLE_MOMENTUM & flags) == 0)
    {
        switch (m_rtype)
        {
        case NONE:
            funct = &AjustClassic;
            break;
        case L1:
            funct = &AjustL1;
            break;
        case L2:
            funct = &AjustL2;
            break;
        default:
            funct = &AjustClassic;
            break;
        }
    }
    else
    {
        switch (m_rtype)
        {
        case NONE:
            functS = &AjustClassicS;
            break;
        case L1:
            functS = &AjustL1S;
            break;
        case L2:
            functS = &AjustL2S;
            break;
        default:
            functS = &AjustClassicS;
            break;
        }
    }

    std::vector<double> antiOF(NonProgressionStopStep);
    std::vector<double> antiOFAverage(NonProgressionStopStep);

    m_costMemory.clear();
    m_costMemory.resize(m_memorySize);

    unsigned int etaCounter(0);
    const unsigned short etaCounterLimit(10);

    m_valueW = ann->m_vW;
    m_biais = ann->m_biais;

    m_momentumSpeed.clear();
    m_momentumSpeed.resize(m_valueW.size());

    // Matrix of delta for a batch
    TOT_D_valueW.resize(m_valueW.size());
    TOT_D_biais.resize(m_biais.size());

    for(int i(0), c(m_biais.size()); i < c; i++)
    {
        m_momentumSpeed[i] = DDMatrix(m_valueW[i].rows(), m_valueW[i].columns(), 0);
        TOT_D_valueW[i] = DDMatrix(m_valueW[i].rows(), m_valueW[i].columns(), 0);
        TOT_D_biais[i] = DDMatrix(m_biais[i].rows(), 1, 0);
    }


    DDMatrix x_train;
    DDMatrix y_train;
    DDMatrix x_test;
    DDMatrix y_test;

    m_data.m_Xtrain.transpose(x_train);
    m_data.m_Ytrain.transpose(y_train);
    m_data.m_Xtest.transpose(x_test);
    m_data.m_Ytest.transpose(y_test);

    int rest(x_train.columns() % m_miniBatchSize);
    int nbrMiniBatch = (x_train.columns() - rest)/m_miniBatchSize;
    int n = m_data.m_nbrSubjectsTrain;
    DDMatrix Xnow;
    DDMatrix Ynow;
    //ANNFile file;
    for(int e(0); e < epoche; e++)
    {
        randomise(x_train, y_train);
        for(int m(0); m < nbrMiniBatch; m++)
        {
            project(Xnow, x_train, brnm::mat::Range(0, x_train.rows()), brnm::mat::Range(m*m_miniBatchSize, (m+1)*m_miniBatchSize));
            project(Ynow, y_train, brnm::mat::Range(0, y_train.rows()), brnm::mat::Range(m*m_miniBatchSize, (m+1)*m_miniBatchSize));
            CalculateDelta(Xnow, Ynow);
            for(int i(0), c(m_biais.size()); i < c; i++)
            {
                if((ENABLE_MOMENTUM & flags) != 0)
                    functS(eta, lambda, mu, n, m_miniBatchSize, m_valueW[i], m_momentumSpeed[i], TOT_D_valueW[i]);
                else
                    funct(eta, lambda, n, m_miniBatchSize, m_valueW[i], TOT_D_valueW[i]);
                TOT_D_biais[i].scale(hmma::Multiplies<double> (), eta);
                m_biais[i] -= TOT_D_biais[i];
            }

            for(int i(0), c(m_biais.size()); i < c; i++)
            {
                m_momentumSpeed[i] = DDMatrix(m_valueW[i].rows(), m_valueW[i].columns(), 0);
                TOT_D_valueW[i] = DDMatrix(m_valueW[i].rows(), m_valueW[i].columns(), 0);
                TOT_D_biais[i] = DDMatrix(m_biais[i].rows(), 1, 0);
            }
        }
        if(rest != 0)
        {
            project(Xnow, x_train, brnm::mat::Range(0, x_train.rows()), brnm::mat::Range(nbrMiniBatch*m_miniBatchSize, x_train.columns()));
            project(Ynow, y_train, brnm::mat::Range(0, y_train.rows()), brnm::mat::Range(nbrMiniBatch*m_miniBatchSize, y_train.columns()));
            CalculateDelta(Xnow, Ynow);
            for(int i(0), c(m_biais.size()); i < c; i++)
            {
                if((ENABLE_MOMENTUM & flags) != 0)
                    functS(eta, lambda, mu, n, rest, m_valueW[i], m_momentumSpeed[i], TOT_D_valueW[i]);
                else
                    funct(eta, lambda, n, rest, m_valueW[i], TOT_D_valueW[i]);
                TOT_D_biais[i].scale(hmma::Multiplies<double> (), eta);
                m_biais[i] -= TOT_D_biais[i];
            }
            for(int i(0), c(m_biais.size()); i < c; i++)
            {
                m_momentumSpeed[i] = DDMatrix(m_valueW[i].rows(), m_valueW[i].columns(), 0);
                TOT_D_valueW[i] = DDMatrix(m_valueW[i].rows(), m_valueW[i].columns(), 0);
                TOT_D_biais[i] = DDMatrix(m_biais[i].rows(), 1, 0);
            }
        }
        // set the ANN
        ann->m_vW = m_valueW;
        ann->m_biais = m_biais;


        // Calculate the cost
        DDMatrix cost(y_test.rows(), 1);
        DDMatrix output;
        DDMatrix out(y_test.rows(), m_testSize);
        for(int i(0); i < m_testSize; i++)
        {
            ann->Run(column(x_test, i), output);
            out.set_column(output.get_column(0).begin(), i);
            cost += m_costFunction(output, column(y_test, i));
        }
        float Rcost(0);
        for(int i(0); i < cost.rows(); i++)
        {
            Rcost += cost(i, 0);
        }
        Rcost /= m_testSize;

        // updating the cost memory
        m_costMemory.push_back(Rcost);
        m_costMemory.erase(m_costMemory.begin());
        int event(0);

        // detecting if the cost oscillate and if yes we divide the learning rate by 2
        if((flags & LEARNING_RATE_VARIATIONS) != 0)
        {
            if(oscillatorDetector())
            {
                if(etaCounter < etaCounterLimit)
                {
                    eta /= 2;
                    etaCounter++;
                    event = 1;
                    for(int i(0); i < m_memorySize; i++)
                        m_costMemory[i] = 0;
                }
            }
        }
        // calculate the accuracy
        float accuracy = Accuracy(out, y_test);

        // set the memory of the last 10 epochs
        antiOF.push_back(accuracy);
        antiOF.erase(antiOF.begin());

        // calculate the average so as to reduce the luck factor
        float average(0);
        for(int i(0); i < NonProgressionStopStep; i++)
            average += antiOF[i];
        average /= NonProgressionStopStep;

        // set the memory of them
        antiOFAverage.push_back(average);
        antiOFAverage.erase(antiOFAverage.begin());

        // if we reached the wanted accuracy on the validation set
        if(accuracy > minAccuracy)
        {
            bool NoProgress(true);
            for(int i(1); i < NonProgressionStopStep; i++)
            {
                if(antiOFAverage[i-1] <= antiOFAverage[i])
                    NoProgress = false;
            }
            if(NoProgress)
            {
                // the ANN begins to get worse so we stop the training
                std::cout << "THE ANN DOESN'T MAKE PROGRESS SINCE " << NonProgressionStopStep << " Epochs" << std::endl;
                break;
            }
        }
        //file.export_(*ann, "tmp.brn");

        if(printtype == DNP)
            continue;
        if(printtype == TAB || printtype == ALL)
            m_perf.push_back(Performances(e, Rcost, accuracy, event));
        if((printtype == MONITOR || printtype == ALL) && (e % printStep) == 0)
            std::cout << "Epoche " << e << " : over" << std::endl << "Cost : " << Rcost << std::endl << "Accuracy : " << accuracy*100 << "%" << std::endl << "*******************************" << std::endl;

}

}

bool ANNTrainer::oscillatorDetector()
{
    // average declared as static variable in order not to have to recompute it every time
    static double average(0);
    average += m_costMemory.back()/m_memorySize;

    //test how many time the cross function in memory cross its average
    int counter(0);
    for(int i(1); i < m_memorySize; i++)
        if((m_costMemory[i-1] > average) != (m_costMemory[i] > average))
            counter++;

    average -= m_costMemory.front()/m_memorySize;
    //if it cross it more than OSCILLATOR_LIMIT time then we consider that the function oscillate
    if(counter > OSCILLATOR_LIMIT)
    {
        average = 0;
        return true;
    }
    return false;
}

void ANNTrainer::CalculateDelta(DDMatrix& Xnow, DDMatrix& Ynow)
{

        std::vector<DDMatrix> Mbiais(m_biais.size());
        std::vector<DDMatrix> zs(m_biais.size());
        std::vector<DDMatrix> activations(m_biais.size()+1);
        for(int i(0); i < m_biais.size(); i++)
        {
            Mbiais[i] = propMat(m_biais[i], Xnow.columns());
        }

        // feedforward
        activations[0] = Xnow;
        for(unsigned int i(0); i < zs.size(); i++)
        {
            zs[i] = (m_valueW[i] * activations[i]) + Mbiais[i];
            activations[i+1] = Rapply_to_all(zs[i], SoftStep());
        }
        DDMatrix delta;

        // back prop
        delta = m_deltaCostDerivative(activations.back(), Ynow, zs.back());
        TOT_D_biais.back() = delta * DDMatrix(delta.columns(), 1, 1/delta.columns());
        TOT_D_valueW.back() = delta * activations[activations.size()-2].transpose();

        for(int i(1); i < zs.size(); i++)
        {
            DDMatrix TW;
            m_valueW[m_valueW.size() - i].transpose(TW);
            apply_to_all(SoftStepPrime(), zs[zs.size() - 1 - i].col_begin(), zs[zs.size() - 1 - i].col_end());
            delta = NORelement_prod( TW * delta, zs[zs.size() - 1 - i]);
            TOT_D_biais[TOT_D_biais.size()-1 -i] = delta * DDMatrix(delta.columns(), 1, 1/delta.columns());
            TOT_D_valueW[TOT_D_valueW.size()-1 -i] = delta * activations[activations.size()-2 -i].transpose();
        }
}
double ANNTrainer::Accuracy(DDMatrix& out, DDMatrix& y)
{
    if(m_arules == NORM)
    {
        double norm(0);
        float counter(0);
        for(int i(0); i < m_testSize; i++)
        {
            DDMatrix save = (column(out, i) - column(y, i));
            norm = save.norm();
            if(norm <= m_MOE)
                counter++;
        }
        return counter/m_testSize;
    }
    if(m_arules == BIGGEST)
    {
        double big(0);
        int counter(0);
        for(int i(0); i < m_testSize; i++)
        {
            for (int j(0); j < out.rows(); j++)
                big = std::abs(out(j, i) - y(j, i)) > big ? std::abs(out(j, i) - y(j, i)) : big;
            if(big <= m_MOE)
                counter++;
        }
        return counter/m_testSize;
    }
    if(m_arules == AVERAGE)
    {
        double avrg(0);
        int counter(0);
        for(int i(0); i < m_testSize; i++)
        {
            for (int j(0); j < out.rows(); j++)
                avrg += std::abs(out(j, i) - y(j, i));
            avrg /= out.rows();
            if(avrg <= m_MOE)
                counter++;
        }
        return counter/m_testSize;
    }
    if(m_arules == ARGMAX)
    {
        float counter(0);
        for(int i(0); i < m_testSize; i++)
        {
            double max1(out(0, i));
            std::size_t idx1(0);
            double max2(y(0, i));
            std::size_t idx2(0);
            for (std::size_t j(1); j < out.rows(); j++)
            {
                max1 = out(j, i) > max1 ? out(j, i) : max1;
                idx1 = out(j, i) == max1 ? j : idx1;

                max2 = y(j, i) > max2 ? y(j, i) : max2;
                idx2 = y(j, i) == max2 ? j : idx2;
            }
            if(idx1 == idx2)
                counter++;
        }
        return counter/m_testSize;
    }
    if(m_arules == PERSONALIZED)
    {
        return m_afunc(out, y, m_MOE);
    }
}

DDMatrix Quadric(DDMatrix& a, DDMatrix y)
{
    hmma::DDMatrix x;
    x = y - a;
    x.scale(hmma::Power<double> (), 2);
    x.scale(hmma::Divides<double> (), 2);
    return x;
}

DDMatrix DeltaQuadricDervative(DDMatrix& x, DDMatrix& y, DDMatrix& z)
{
    DDMatrix result = NRelement_prod(x - y, apply_to_all(z, SoftStepPrime()));
    return result;
}

DDMatrix CrossEntropy(DDMatrix& x, DDMatrix y)
{
    DDMatrix _2hs(y.rows(), 1);
    _2hs -= NRelement_prod(DDMatrix(y.rows(), 1, 1) - y, apply_to_all(DDMatrix(x.rows(), 1, 1) - x, Log())) + element_prod(y, me_apply_to_all(x, Log()));
    return _2hs;
}

DDMatrix DeltaCrossEntropyDerivative(DDMatrix& x, DDMatrix& y, DDMatrix& z)
{
    DDMatrix result = x - y;
    return result;
}

// set the weights without momentum
void AjustClassic(double const& eta, double const& lambda, double const& n, double const& m, DDMatrix& w, DDMatrix& delta)
{
    delta.scale(hmma::Multiplies<double> (), eta/m);
    w -= delta;
}

void AjustL1(double const& eta, double const& lambda, double const& n, double const& m, DDMatrix& w, DDMatrix& delta)
{
    DDMatrix L1 = apply_to_all(w, Sgn());
    L1.scale(hmma::Multiplies<double> (), (eta*lambda)/n); // L1
    delta.scale(hmma::Multiplies<double> (), eta/m);

    w -= (L1 + delta);
}

void AjustL2(double const& eta, double const& lambda, double const& n, double const& m, DDMatrix& w, DDMatrix& delta)
{
    w.scale(hmma::Multiplies<double> (), 1 - (eta*lambda)/n); //L2
    delta.scale(hmma::Multiplies<double> (), eta/m);
    w -= delta;
}

// set the weights with momentum
void AjustClassicS(double const& eta, double const& lambda, double const& mu, double const& n, double const& m, DDMatrix& w, DDMatrix& momentum, DDMatrix& delta)
{
    momentum.scale(hmma::Multiplies<double> (), mu);
    delta.scale(hmma::Multiplies<double> (), eta/m);
    momentum -= delta;
    w += momentum;
}
void AjustL1S(double const& eta, double const& lambda, double const& mu, double const& n, double const& m, DDMatrix& w, DDMatrix& momentum, DDMatrix& delta)
{
    DDMatrix L1 = apply_to_all(w, Sgn());
    L1.scale(hmma::Multiplies<double> (), (eta*lambda)/n); // L1
    delta.scale(hmma::Multiplies<double> (), eta/m);
    momentum.scale(hmma::Multiplies<double> (), mu);
    momentum -= (L1 + delta);
    w += momentum;
}

void AjustL2S(double const& eta, double const& lambda, double const& mu, double const& n, double const& m, DDMatrix& w, DDMatrix& momentum, DDMatrix& delta)
{
    momentum.scale(hmma::Multiplies<double> (), mu);
    DDMatrix tmpW = w;
    tmpW.scale(hmma::Multiplies<double> (), eta*(lambda/n));
    delta.scale(hmma::Multiplies<double> (), eta/n);
    momentum -= (delta + tmpW);
    w += momentum;
}

bool randomise(DDMatrix &x, DDMatrix &y)
{
    DDMatrix newX(x.rows(), x.columns());
    DDMatrix newY(y.rows(), y.columns());
    std::vector<int> idx = randomSequence(0, x.columns());

    for(int i(0), c(x.columns()); i < c; i++)
    {
        newX.set_column(x.get_column(i).begin(), i);
        newY.set_column(y.get_column(i).begin(), i);
    }

    x = newX;
    y = newY;

    return true;
}

std::istream& operator>> (std::istream& flux, Performances& perfs)
{
    flux >> perfs.Epoche >> perfs.Cost >> perfs.Accuracy >> perfs.m_event;
    return flux;
}

std::ostream &operator<<(std::ostream &flux, Performances const& perf)
{
    flux << perf.Epoche << " " << perf.Cost << " " << perf.Accuracy << " " << perf.m_event << " ";
    return flux;
}
