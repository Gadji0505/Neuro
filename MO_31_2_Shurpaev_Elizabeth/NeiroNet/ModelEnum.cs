namespace MO_31_2_Shurpaev_Elizabeth.NeiroNet
{
    enum MemoryMode // режим работы памяти (геттеры, сеттеры)
    {
        GET,        // считывание памяти
        SET,        // сохранение памяти
        INIT        // инициализация памяти
    }

    enum NeuronType // тип нейрона
    {
        Hidden,     // скрытый
        Output      // выходной
    }

    enum NetworkMode// типа нейрона
    {
        Train, // обучение
        Test,  // тестовый
        Demo   // распознавание
    }
}