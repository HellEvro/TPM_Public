// Простой тест для проверки синтаксиса
class TestBotsManager {
    constructor() {
        console.log('TestBotsManager created');
    }
    
    test() {
        console.log('Test method called');
    }
}

// Экспортируем класс глобально
window.TestBotsManager = TestBotsManager;

console.log('Test file loaded successfully');
